from kipr_array import *
from . import unicode_simplify
from . import sys_utils as sys

text = type('k_module', (object,), {'normalization_map': unicode_simplify.normalize})()
