# Package initialization
from .metrics import SegmentationMetrics, PairedTTest

try:
    from .visualization import ResultVisualizer
    __all__ = ['SegmentationMetrics', 'PairedTTest', 'ResultVisualizer']
except ImportError:
    __all__ = ['SegmentationMetrics', 'PairedTTest']
