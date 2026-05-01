# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class SegmentationMetrics:
    """
    Performance metrics for segmentation evaluation
    As defined in Chapter 4 (Section 4.3.1)
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def overall_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Overall Accuracy (Equation 4.1)
        Proportion of correctly classified pixels
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    def average_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Average Accuracy (Equation 4.2)
        Mean of per-class accuracies
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        # Ignore classes with no samples
        per_class_acc = per_class_acc[~np.isnan(per_class_acc)]
        return np.mean(per_class_acc)
    
    def mean_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        mean Intersection over Union (Equation 4.3)
        Average overlap between predicted and ground truth regions
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        intersection = cm.diagonal()
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        
        # IoU per class
        iou = intersection / union
        iou = iou[~np.isnan(iou)]
        
        return np.mean(iou)
    
    def per_class_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Per-class F1 score (Equation 4.4)
        Harmonic mean of precision and recall
        """
        return f1_score(y_true, y_pred, labels=range(self.num_classes), average=None)
    
    def macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Macro-averaged F1 (Equation 4.5)
        Equal weight to all classes regardless of size
        """
        return f1_score(y_true, y_pred, labels=range(self.num_classes), average='macro')
    
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all metrics and return as dictionary"""
        return {
            'overall_accuracy': self.overall_accuracy(y_true, y_pred),
            'average_accuracy': self.average_accuracy(y_true, y_pred),
            'mean_iou': self.mean_iou(y_true, y_pred),
            'macro_f1': self.macro_f1(y_true, y_pred),
            'per_class_f1': self.per_class_f1(y_true, y_pred)
        }


class PairedTTest:
    """
    Paired t-test for statistical significance (Section 4.3.2)
    """
    
    @staticmethod
    def critical_value(alpha: float, df: int = 9) -> float:
        """
        Critical values for one-tailed paired t-test (df = n-1 = 9)
        Based on Table 4.2 in the thesis
        """
        critical_values = {
            0.01: 2.821,  # 99% confidence
            0.05: 1.833,  # 95% confidence
            0.10: 1.383   # 90% confidence
        }
        return critical_values.get(alpha, 1.833)
    
    @staticmethod
    def test(baseline_scores: np.ndarray, enhanced_scores: np.ndarray, alpha: float = 0.05):
        """
        Perform paired t-test
        
        Returns:
            t_statistic: Calculated t-value
            significant: Whether difference is significant at given alpha
            improvement: Mean improvement
        """
        differences = enhanced_scores - baseline_scores
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        t_statistic = mean_diff / (std_diff / np.sqrt(n))
        
        critical_t = PairedTTest.critical_value(alpha, df=n-1)
        significant = t_statistic > critical_t
        
        return {
            't_statistic': t_statistic,
            'critical_value': critical_t,
            'significant': significant,
            'mean_improvement': mean_diff,
            'confidence': 1 - alpha
        }