# src/optimization/watershed_ga.py
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import watershed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch

class WatershedGA:
    """
    Genetic Algorithm for watershed pipeline optimization
    Co-optimizes: n_PCA, sigma, tau_grad, s_min, classifier
    """
    
    def __init__(self, 
                 population_size=50,
                 max_generations=100,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elitism_rate=0.1):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
        # Parameter bounds
        self.bounds = {
            'n_pca': (1, 10),        # Number of PCA components
            'sigma': (0.1, 5.0),     # Gaussian smoothing sigma
            'tau_grad': (0.01, 0.99), # Gradient threshold
            's_min': (1, 500),       # Minimum region size
            'classifier': (0, 2)     # 0: RF, 1: SVM, 2: k-NN
        }
    
    def encode_chromosome(self, params):
        """Encode parameters as chromosome vector"""
        return np.array([
            params['n_pca'],
            params['sigma'],
            params['tau_grad'],
            params['s_min'],
            params['classifier']
        ])
    
    def decode_chromosome(self, chromosome):
        """Decode chromosome to parameter dictionary"""
        return {
            'n_pca': int(np.clip(chromosome[0], self.bounds['n_pca'][0], self.bounds['n_pca'][1])),
            'sigma': np.clip(chromosome[1], self.bounds['sigma'][0], self.bounds['sigma'][1]),
            'tau_grad': np.clip(chromosome[2], self.bounds['tau_grad'][0], self.bounds['tau_grad'][1]),
            's_min': int(np.clip(chromosome[3], self.bounds['s_min'][0], self.bounds['s_min'][1])),
            'classifier': int(np.clip(chromosome[4], self.bounds['classifier'][0], self.bounds['classifier'][1]))
        }
    
    def fitness(self, chromosome, data, labels):
        """
        Fitness function as defined in Equation 6.5
        f = 0.55 * OA + 0.45 * AA - coverage_penalty - smoothness_penalty
        """
        params = self.decode_chromosome(chromosome)
        
        # Run watershed segmentation
        segmentation = self.run_watershed(data, params)
        
        # Compute metrics
        oa, aa = self.compute_accuracy(segmentation, labels)
        
        # Compute penalties
        coverage_penalty = self.coverage_penalty(segmentation)
        smoothness_penalty = self.smoothness_penalty(segmentation)
        
        # Final fitness
        fitness_value = 0.55 * oa + 0.45 * aa - coverage_penalty - smoothness_penalty
        
        return fitness_value
    
    def run_watershed(self, data, params):
        """
        Execute marker-controlled watershed pipeline
        
        Steps:
        1. PCA dimensionality reduction
        2. Gaussian smoothing
        3. Gradient computation
        4. Marker generation (internal + external)
        5. Watershed flooding
        6. Region classification
        """
        from sklearn.decomposition import PCA
        
        # Step 1: PCA dimensionality reduction
        h, w, b = data.shape
        data_flat = data.reshape(-1, b)
        pca = PCA(n_components=params['n_pca'])
        pca_result = pca.fit_transform(data_flat)
        pca_image = pca_result.reshape(h, w, params['n_pca'])
        
        # Use first principal component for gradient computation
        img = pca_image[:, :, 0]
        
        # Step 2: Gaussian smoothing
        img_smooth = gaussian_filter(img, sigma=params['sigma'])
        
        # Step 3: Gradient computation (Sobel)
        from scipy.ndimage import sobel
        gradient = np.abs(sobel(img_smooth, axis=0)) + np.abs(sobel(img_smooth, axis=1))
        
        # Step 4: Marker generation (Otsu's thresholding)
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(gradient)
        markers = gradient > (thresh * params['tau_grad'])
        
        # Step 5: Watershed segmentation
        from scipy import ndimage as ndi
        markers_labeled = ndi.label(markers)[0]
        segmentation = watershed(gradient, markers_labeled, mask=img > 0)
        
        # Step 6: Remove small regions
        for region_id in range(1, segmentation.max() + 1):
            if np.sum(segmentation == region_id) < params['s_min']:
                segmentation[segmentation == region_id] = 0
        
        return segmentation
    
    def compute_accuracy(self, segmentation, labels):
        """Compute Overall Accuracy and Average Accuracy"""
        # Convert segmentation regions to class labels
        region_labels = self.classify_regions(segmentation, labels)
        
        # Compute OA and AA
        correct = np.sum(region_labels == labels)
        total = len(labels)
        oa = correct / total
        
        # Per-class accuracy
        unique_classes = np.unique(labels)
        class_accuracies = []
        for cls in unique_classes:
            mask = labels == cls
            if np.sum(mask) > 0:
                cls_correct = np.sum(region_labels[mask] == cls)
                class_accuracies.append(cls_correct / np.sum(mask))
        
        aa = np.mean(class_accuracies) if class_accuracies else 0
        
        return oa, aa
    
    def classify_regions(self, segmentation, labels):
        """Classify segments using chosen classifier"""
        region_ids = np.unique(segmentation)[1:]  # Skip background
        region_labels = np.zeros_like(segmentation)
        
        for region_id in region_ids:
            mask = segmentation == region_id
            # Majority voting
            region_class = np.bincount(labels[mask].astype(int)).argmax()
            region_labels[mask] = region_class
        
        return region_labels
    
    def coverage_penalty(self, segmentation):
        """Penalize solutions that leave too many unlabeled pixels"""
        labeled_pixels = np.sum(segmentation > 0)
        total_pixels = segmentation.size
        coverage_ratio = labeled_pixels / total_pixels
        
        if coverage_ratio < 0.95:
            return 0.1 * (0.95 - coverage_ratio)
        return 0
    
    def smoothness_penalty(self, segmentation):
        """Penalize fragmented segmentations"""
        n_regions = len(np.unique(segmentation)) - 1
        max_regions = 500
        
        if n_regions > max_regions:
            return min(0.2, 0.001 * (n_regions - max_regions))
        return 0
    
    def evolve(self, data, labels, verbose=False):
        """Main GA evolution loop (Algorithm 2)"""
        # Initialize population
        population = self._initialize_population()
        best_chromosome = None
        best_fitness = -np.inf
        no_improve_count = 0
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(ind, data, labels) for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if verbose:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
            
            # Stopping condition
            if no_improve_count >= 10:
                if verbose:
                    print(f"Early stopping at generation {generation}")
                break
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            n_elite = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_scores)[-n_elite:]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # Tournament selection and crossover
            while len(new_population) < self.population_size:
                parents = self._tournament_selection(population, fitness_scores, k=3)
                
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parents[0], parents[1])
                else:
                    child = parents[0].copy()
                
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return self.decode_chromosome(best_chromosome), best_fitness
    
    def _initialize_population(self):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = np.array([
                np.random.uniform(*self.bounds['n_pca']),
                np.random.uniform(*self.bounds['sigma']),
                np.random.uniform(*self.bounds['tau_grad']),
                np.random.uniform(*self.bounds['s_min']),
                np.random.randint(*self.bounds['classifier'])
            ])
            population.append(individual)
        return population
    
    def _tournament_selection(self, population, fitness_scores, k=3):
        """Tournament selection"""
        selected = []
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), size=k, replace=False)
            winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
            selected.append(population[winner_idx].copy())
        return selected
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def _mutate(self, individual):
        """Mutation with probability mutation_rate"""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                bounds_key = list(self.bounds.keys())[i]
                if i == 4:  # classifier (integer)
                    individual[i] = np.random.randint(*self.bounds[bounds_key])
                else:
                    individual[i] = np.random.uniform(*self.bounds[bounds_key])
        return individual
