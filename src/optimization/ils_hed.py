# src/optimization/ils_hed.py
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from typing import Dict, List, Tuple, Any

class ClassicalDetectorConfig:
    """Configuration space for classical edge detectors"""
    
    def __init__(self):
        # Canny configurations: 9 variants
        self.canny_thresholds = [(30, 90), (50, 150), (75, 225)]  # 1:3 ratio
        self.canny_apertures = [3, 5, 7]
        self.canny_configs = list(product(self.canny_thresholds, self.canny_apertures))
        
        # Sobel configurations: 4 variants
        self.sobel_kernels = [3, 5, 7, 9]
        
        # Laplacian configurations: 4 variants
        self.laplacian_kernels = [3, 5, 7, 9]
        
        # Gabor configurations: 36 variants
        self.gabor_orientations = [4, 8, 12]
        self.gabor_wavelengths = [4, 8, 16]
        self.gabor_bandwidths = [0.5, 1.0, 2.0]
        self.gabor_configs = list(product(self.gabor_orientations, 
                                         self.gabor_wavelengths, 
                                         self.gabor_bandwidths))
    
    def get_total_configs(self) -> int:
        return (len(self.canny_configs) * 
                len(self.sobel_kernels) * 
                len(self.laplacian_kernels) * 
                len(self.gabor_configs))

class IterativeLocalSearch:
    """
    Iterative Local Search for hyper-heuristic feature selection
    Burke et al. (2013) definition of hyper-heuristics
    """
    
    def __init__(self, max_iterations=50, patience=10):
        self.max_iterations = max_iterations
        self.patience = patience
        
    def search(self, validation_data, scorer, initial_config=None):
        """
        Main ILS optimization loop
        
        Args:
            validation_data: Validation dataset
            scorer: Scoring function (negative weighted BCE)
            initial_config: Starting configuration
        """
        best_config = initial_config or self._random_config()
        best_score = self._evaluate(best_config, validation_data, scorer)
        
        no_improve = 0
        
        for iteration in range(self.max_iterations):
            # Perturb current best
            perturbed = self._perturb(best_config)
            
            # Evaluate perturbed configuration
            score = self._evaluate(perturbed, validation_data, scorer)
            
            if score > best_score:
                best_config = perturbed
                best_score = score
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.patience:
                break
        
        return best_config, best_score
    
    def _random_config(self) -> Dict:
        """Generate random configuration of classical detectors"""
        detector_configs = ClassicalDetectorConfig()
        
        return {
            'canny': np.random.choice(len(detector_configs.canny_configs)),
            'sobel': np.random.choice(len(detector_configs.sobel_kernels)),
            'laplacian': np.random.choice(len(detector_configs.laplacian_kernels)),
            'gabor': np.random.choice(len(detector_configs.gabor_configs))
        }
    
    def _perturb(self, config: Dict) -> Dict:
        """Apply perturbation: Add, Remove, or Swap"""
        perturb_type = np.random.choice(['add', 'remove', 'swap'])
        new_config = config.copy()
        
        if perturb_type == 'swap':
            # Replace a detector configuration with a different variant
            detector = np.random.choice(list(new_config.keys()))
            detector_configs = ClassicalDetectorConfig()
            max_val = {
                'canny': len(detector_configs.canny_configs),
                'sobel': len(detector_configs.sobel_kernels),
                'laplacian': len(detector_configs.laplacian_kernels),
                'gabor': len(detector_configs.gabor_configs)
            }
            new_val = np.random.randint(0, max_val[detector])
            new_config[detector] = new_val
        
        return new_config
    
    def _evaluate(self, config: Dict, validation_data, scorer) -> float:
        """Evaluate configuration using scoring function"""
        if validation_data is None:
            return scorer(config, None, None)

        # Extract classical features based on config
        classical_features = self._extract_classical_features(validation_data, config)

        # Get HED predictions
        hed_outputs = self._get_hed_predictions(validation_data)

        if classical_features is None or hed_outputs is None:
            return scorer(config, None, None)

        # Compute fused score
        score = scorer(config, classical_features, hed_outputs)

        return score

    def _extract_classical_features(self, data, config):
        """Extract classical edge features based on configuration"""
        import cv2

        if data is None:
            return {}

        detector_configs = ClassicalDetectorConfig()

        # Normalise data to uint8 grayscale for OpenCV
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                # Multi-band: use mean across bands
                gray = data.mean(axis=-1)
            else:
                gray = data.copy()
            gray = gray.astype(np.float32)
            g_min, g_max = gray.min(), gray.max()
            if g_max > g_min:
                gray = ((gray - g_min) / (g_max - g_min) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)
        else:
            return {}

        features = {}

        # --- Canny ---
        canny_configs = detector_configs.canny_configs
        if canny_configs:
            canny_idx = int(config.get('canny', 0)) % len(canny_configs)
            (low_thresh, high_thresh), aperture = canny_configs[canny_idx]
            features['canny'] = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=aperture)

        # --- Sobel ---
        sobel_kernels = detector_configs.sobel_kernels
        if sobel_kernels:
            sobel_idx = int(config.get('sobel', 0)) % len(sobel_kernels)
            ksize = sobel_kernels[sobel_idx]
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            features['sobel'] = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # --- Laplacian ---
        laplacian_kernels = detector_configs.laplacian_kernels
        if laplacian_kernels:
            lap_idx = int(config.get('laplacian', 0)) % len(laplacian_kernels)
            lap_ksize = laplacian_kernels[lap_idx]
            features['laplacian'] = cv2.Laplacian(gray, cv2.CV_64F, ksize=lap_ksize)

        # --- Gabor filter bank ---
        gabor_configs = detector_configs.gabor_configs
        if gabor_configs:
            gabor_idx = int(config.get('gabor', 0)) % len(gabor_configs)
            n_orientations, wavelength, bandwidth = gabor_configs[gabor_idx]
            gabor_responses = []
            for angle_idx in range(n_orientations):
                theta = np.pi * angle_idx / n_orientations
                sigma = wavelength / np.pi * np.sqrt(np.log(2) / 2) * (2 ** bandwidth + 1) / (2 ** bandwidth - 1)
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    sigma,
                    theta,
                    wavelength,
                    bandwidth,
                    0,
                    ktype=cv2.CV_32F,
                )
                response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
                gabor_responses.append(response)
            features['gabor'] = np.stack(gabor_responses, axis=-1)

        return features

    def _get_hed_predictions(self, data):
        """Get HED predictions (frozen model)"""
        import torch

        if data is None:
            return None

        if not isinstance(data, np.ndarray):
            return None

        from src.models.hed import HolisticallyNestedEdgeDetection

        # Initialise a frozen HED model (lazy, no pretrained weights required at test time)
        hed_model = HolisticallyNestedEdgeDetection(pretrained=False)
        hed_model.eval()
        for param in hed_model.parameters():
            param.requires_grad_(False)

        # Prepare a 3-channel float tensor (H, W, 3) → (1, 3, H, W)
        if data.ndim == 2:
            img = np.stack([data] * 3, axis=-1)
        elif data.ndim == 3 and data.shape[2] >= 3:
            img = data[:, :, :3]
        elif data.ndim == 3:
            img = np.concatenate(
                [data] + [data[:, :, -1:]] * (3 - data.shape[2]), axis=-1
            )
        else:
            return None

        img = img.astype(np.float32)
        i_min, i_max = img.min(), img.max()
        if i_max > i_min:
            img = (img - i_min) / (i_max - i_min)

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            side_outputs = hed_model(tensor)
            fused = torch.sigmoid(side_outputs[-1]).squeeze().cpu().numpy()

        return fused

class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Weighted binary cross-entropy loss for edge detection
    As defined in Equation 5.6 of the thesis
    """
    
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted edge map F
            targets: Ground truth edge map Y
        """
        batch_size = predictions.shape[0]
        loss = 0
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            N = pred.numel()
            beta = (target.sum() / N).clamp(min=self.epsilon, max=1 - self.epsilon)
            
            w_pos = 1.0 / (beta + self.epsilon)
            w_neg = 1.0 / (1 - beta + self.epsilon)
            
            # Weighted binary cross-entropy
            pos_loss = -w_pos * target * torch.log(pred + self.epsilon)
            neg_loss = -w_neg * (1 - target) * torch.log(1 - pred + self.epsilon)
            
            loss_i = (pos_loss + neg_loss).mean()
            loss += loss_i
        
        return loss / batch_size