# Package initialization
try:
    from .ils_hed import IterativeLocalSearch, WeightedBinaryCrossEntropyLoss
    _ils_available = True
except ImportError:
    _ils_available = False

try:
    from .watershed_ga import WatershedGA
    _ws_available = True
except ImportError:
    _ws_available = False

from .degree_ga import DEGAHybrid

__all__ = ['DEGAHybrid']
if _ils_available:
    __all__ += ['IterativeLocalSearch', 'WeightedBinaryCrossEntropyLoss']
if _ws_available:
    __all__ += ['WatershedGA']
