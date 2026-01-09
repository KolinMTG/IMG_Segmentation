"""
Class-specific detection modules for satellite image segmentation.

Each module implements a specialized detection algorithm for one semantic class.
"""

# ! TO ADD NEW CLASS MODULES, IMPORT THEM HERE !
# 1. create a treatement file in this module, with the name of the class (e.g., water.py)
# 2. implement the process_<class> function in that file
# 3. import the function here
# 4. add the function name to the __all__ list below
# 5. update the cste.py file if necessary to include the new class info (e.g.,id, name, color)

# Import necessary modules
from .field import process_field
from .building import process_building
from .woodland import process_woodland
from .water import process_water
from .road import process_road


# Defines what gets exported when someone does from classes import *
__all__ = [
    'process_field',
    'process_building',
    'process_woodland',
    'process_water',
    'process_road',
]
