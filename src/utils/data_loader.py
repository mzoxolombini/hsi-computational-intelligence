# src/utils/data_loader.py
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class HyperspectralDataLoader:
    """
    Load benchmark hyperspectral datasets
    Indian Pines, Pavia University, Salinas Valley, Houston, Botswana
    """
    
    def __init__(self, data_dir: str = './data/raw'):
        self.data_dir = data_dir
        
        # Dataset configurations
        self.datasets = {
            'indian_pines': {
                'filename': 'IndianPines.mat',
                'dimensions': (145, 145, 200),
                'num_classes': 16
            },
            'pavia_university': {
                'filename': 'PaviaU.mat',
                'dimensions': (610, 340, 103),
                'num_classes': 9
            },
            'salinas_valley': {
                'filename': 'Salinas.mat',
                'dimensions': (512, 217, 204),
                'num_classes': 16
            },
            'houston': {
                'filename': 'Houston.mat',
                'dimensions': (349, 1905, 144),
                'num_classes': 15
            },
            'botswana': {
                'filename': 'Botswana.mat',
                'dimensions': (1476, 256, 145),
                'num_classes': 14
            }
        }
    
    def load_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load hyperspectral data and ground truth labels
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            data: Hyperspectral cube (H, W, B)
            labels: Ground truth labels (H, W)
        """
        config = self.datasets.get(dataset_name.lower())
        if not config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        filepath = f"{self.data_dir}/{config['filename']}"
        
        try:
            mat_data = sio.loadmat(filepath)
            
            # Dataset-specific loading
            if dataset_name.lower() == 'indian_pines':
                data = mat_data['indian_pines_corrected']
                labels = mat_data['indian_pines_gt']
            elif dataset_name.lower() == 'pavia_university':
                data = mat_data['paviaU']
                labels = mat_data['paviaU_gt']
            elif dataset_name.lower() == 'salinas_valley':
                data = mat_data['salinas_corrected']
                labels = mat_data['salinas_gt']
            elif dataset_name.lower() == 'houston':
                data = mat_data['houston_data']
                labels = mat_data['houston_gt']
            elif dataset_name.lower() == 'botswana':
                data = mat_data['botswana_data']
                labels = mat_data['botswana_gt']
            else:
                data = mat_data['data']
                labels = mat_data['labels']
            
            return data.astype(np.float32), labels.astype(np.int32)
            
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found. Please download the dataset.")
            return None, None
    
    def split_data(self, labels: np.ndarray, train_ratio: float = 0.7) -> Dict:
        """
        Stratified split into training and testing sets (70/30 protocol)
        
        Args:
            labels: Ground truth labels
            train_ratio: Proportion of labeled pixels for training
            
        Returns:
            Dictionary with train_mask, test_mask, and sample indices
        """
        # Get labeled pixels
        labeled_mask = labels > 0
        labeled_indices = np.where(labeled_mask)
        pixel_labels = labels[labeled_mask]
        
        # Stratified split
        train_idx, test_idx = train_test_split(
            np.arange(len(pixel_labels)),
            train_size=train_ratio,
            stratify=pixel_labels,
            random_state=42
        )
        
        # Create masks
        train_mask = np.zeros_like(labels, dtype=bool)
        test_mask = np.zeros_like(labels, dtype=bool)
        
        train_mask[labeled_indices[0][train_idx], labeled_indices[1][train_idx]] = True
        test_mask[labeled_indices[0][test_idx], labeled_indices[1][test_idx]] = True
        
        return {
            'train_mask': train_mask,
            'test_mask': test_mask,
            'train_indices': train_idx,
            'test_indices': test_idx,
            'train_labels': pixel_labels[train_idx],
            'test_labels': pixel_labels[test_idx]
        }
    
    def get_pixel_spectra(self, data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Extract pixel spectra at given indices
        
        Args:
            data: Hyperspectral cube (H, W, B)
            indices: Array of (row, col) indices
            
        Returns:
            Spectra matrix (N_samples, B)
        """
        spectra = []
        for i, j in indices:
            spectra.append(data[i, j, :])
        return np.array(spectra)