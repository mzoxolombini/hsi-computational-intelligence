# Package initialization
try:
    from .hed import HolisticallyNestedEdgeDetection
    __all__ = ['HolisticallyNestedEdgeDetection']
except ImportError:
    __all__ = []
