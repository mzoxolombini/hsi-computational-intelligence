# src/optimization/degree_ga.py
import numpy as np
from typing import Tuple, List

class DEGAHybrid:
    """
    DE-GA Hybrid for Multilevel Thresholding
    Combines Differential Evolution for exploration with GA for exploitation
    """
    
    def __init__(self,
                 num_thresholds: int,
                 population_size: int = 50,
                 max_generations: int = 100,
                 F: float = 0.5,      # DE mutation factor
                 CR: float = 0.8,     # DE crossover rate
                 pc: float = 0.8,     # GA crossover rate
                 pm: float = 0.1):    # GA mutation rate
        
        self.num_thresholds = num_thresholds
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.pc = pc
        self.pm = pm
        
        self.bounds = (0, 255)  # Gray level bounds for 8-bit images
    
    def otsu_fitness(self, thresholds: np.ndarray, histogram: np.ndarray) -> float:
        """
        Otsu's between-class variance fitness function (Equation 7.1)
        
        Args:
            thresholds: Array of k threshold values
            histogram: Image histogram (256 bins)
        """
        thresholds = np.sort(thresholds)
        
        # Normalize histogram to get probabilities
        total_pixels = np.sum(histogram)
        prob = histogram / total_pixels
        
        # Compute class probabilities and means
        classes = np.zeros(len(thresholds) + 1)
        class_means = np.zeros(len(thresholds) + 1)
        class_variances = np.zeros(len(thresholds) + 1)
        
        # Build class boundaries
        boundaries = [-1] + thresholds.tolist() + [255]
        
        for i in range(len(classes)):
            pixel_range = range(boundaries[i] + 1, boundaries[i + 1] + 1)
            class_pixels = [p for p in pixel_range if p < len(prob)]
            class_probs = prob[class_pixels]
            classes[i] = np.sum(class_probs)
            
            if classes[i] > 0:
                class_means[i] = np.sum(np.array(class_pixels) * class_probs) / classes[i]
        
        # Total mean
        total_mean = np.sum(np.arange(len(prob)) * prob)
        
        # Between-class variance
        between_var = np.sum(classes * (class_means - total_mean)**2)
        
        return between_var
    
    def initialize_population(self) -> np.ndarray:
        """Initialize population with random threshold vectors"""
        population = []
        for _ in range(self.population_size):
            # Generate sorted random thresholds
            thresholds = np.sort(np.random.randint(self.bounds[0], self.bounds[1] + 1, 
                                                   size=self.num_thresholds))
            population.append(thresholds)
        return np.array(population)
    
    def differential_mutation(self, population: np.ndarray, idx: int) -> np.ndarray:
        """
        Differential Evolution mutation (Equation 7.4)
        v_i = x_{r1} + F * (x_{r2} - x_{r3})
        """
        # Select three distinct random individuals
        available = [i for i in range(len(population)) if i != idx]
        r1, r2, r3 = np.random.choice(available, size=3, replace=False)
        
        # Compute mutant vector
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        # Clip to bounds
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        
        # Ensure thresholds are sorted
        mutant = np.sort(mutant)
        
        return mutant
    
    def de_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        DE crossover operation (Equation 7.5)
        """
        trial = target.copy()
        j_rand = np.random.randint(0, self.num_thresholds)
        
        for j in range(self.num_thresholds):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return np.sort(trial)
    
    def ga_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover for GA phase"""
        point = np.random.randint(1, self.num_thresholds)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return np.sort(child1), np.sort(child2)
    
    def ga_mutate(self, individual: np.ndarray) -> np.ndarray:
        """GA mutation with Gaussian perturbation"""
        if np.random.random() < self.pm:
            idx = np.random.randint(0, self.num_thresholds)
            perturbation = np.random.normal(0, 10)  # σ = 10 as in thesis
            individual[idx] = np.clip(individual[idx] + perturbation, self.bounds[0], self.bounds[1])
            return np.sort(individual)
        return individual
    
    def optimize(self, histogram: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Main DE-GA optimization loop (Algorithm 3)
        
        Args:
            histogram: Image histogram (256 bins)
            verbose: Print progress if True
            
        Returns:
            optimal_thresholds: Best threshold vector found
            best_fitness: Best fitness value
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitness = np.array([self.otsu_fitness(ind, histogram) for ind in population])
        
        best_idx = np.argmax(fitness)
        best_thresholds = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        no_improve_count = 0
        
        for generation in range(self.max_generations):
            # === DE Phase ===
            trial_population = []
            trial_fitness = []
            
            for i in range(self.population_size):
                # DE mutation and crossover
                mutant = self.differential_mutation(population, i)
                trial = self.de_crossover(population[i], mutant)
                trial_fit = self.otsu_fitness(trial, histogram)
                
                # Selection: keep better solution
                if trial_fit > fitness[i]:
                    trial_population.append(trial)
                    trial_fitness.append(trial_fit)
                else:
                    trial_population.append(population[i].copy())
                    trial_fitness.append(fitness[i])
            
            population = np.array(trial_population)
            fitness = np.array(trial_fitness)
            
            # === GA Phase ===
            # Sort by fitness
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Elitism: keep top 10%
            n_elite = max(1, int(0.1 * self.population_size))
            new_population = population[:n_elite].tolist()
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parents_idx = np.random.choice(range(min(20, self.population_size)), 
                                               size=2, replace=False)
                parent1 = population[parents_idx[0]]
                parent2 = population[parents_idx[1]]
                
                if np.random.random() < self.pc:
                    child1, child2 = self.ga_crossover(parent1, parent2)
                    child1 = self.ga_mutate(child1)
                    child2 = self.ga_mutate(child2)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(self.ga_mutate(parent1.copy()))
                    if len(new_population) < self.population_size:
                        new_population.append(self.ga_mutate(parent2.copy()))
            
            population = np.array(new_population[:self.population_size])
            fitness = np.array([self.otsu_fitness(ind, histogram) for ind in population])
            
            # Update best
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_thresholds = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
            
            # Early stopping (20 generations without improvement)
            if no_improve_count >= 20:
                if verbose:
                    print(f"Early stopping at generation {generation}")
                break
        
        return best_thresholds, best_fitness