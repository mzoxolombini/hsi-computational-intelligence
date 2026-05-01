# src/evaluation/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional

class ResultVisualizer:
    """Visualization tools for segmentation results"""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               xlabel='Predicted Label', ylabel='True Label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_segmentation_comparison(original: np.ndarray, 
                                      baseline_pred: np.ndarray,
                                      enhanced_pred: np.ndarray,
                                      ground_truth: np.ndarray,
                                      save_path: Optional[str] = None):
        """Plot side-by-side segmentation comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image (first principal component)
        if original.ndim == 3:
            original_display = original[:, :, 0]
        else:
            original_display = original
        
        axes[0, 0].imshow(original_display, cmap='gray')
        axes[0, 0].set_title('Original Image (PC1)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(baseline_pred, cmap='tab20', interpolation='none')
        axes[0, 1].set_title('Baseline Segmentation')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(enhanced_pred, cmap='tab20', interpolation='none')
        axes[1, 0].set_title('Enhanced Segmentation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ground_truth, cmap='tab20', interpolation='none')
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_convergence_curves(curves: Dict[str, List[float]], 
                                 save_path: Optional[str] = None):
        """Plot convergence curves for optimization algorithms"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['blue', 'green', 'red']
        for (name, values), color in zip(curves.items(), colors):
            ax.plot(values, label=name, color=color, linewidth=2)
        
        ax.set_xlabel('Iteration / Generation')
        ax.set_ylabel('Fitness Value')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_class_wise_performance(class_names: List[str],
                                     baseline_f1: List[float],
                                     enhanced_f1: List[float],
                                     save_path: Optional[str] = None):
        """Plot class-wise F1 comparison"""
        x = np.arange(len(class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, enhanced_f1, width, label='Enhanced', color='green', alpha=0.7)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_failure_cases(original: np.ndarray,
                           segmentation: np.ndarray,
                           failure_type: str,
                           save_path: Optional[str] = None):
        """Visualize failure cases as in Figures 9.1-9.3"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if original.ndim == 3:
            display_img = original[:, :, 0]
        else:
            display_img = original
        
        axes[0].imshow(display_img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(segmentation, cmap='tab20', interpolation='none')
        axes[1].set_title('Segmentation Result')
        axes[1].axis('off')
        
        axes[2].text(0.5, 0.5, f'Failure Case:\n{failure_type}',
                    ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
        axes[2].axis('off')
        
        plt.suptitle(f'Failure Analysis: {failure_type}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()