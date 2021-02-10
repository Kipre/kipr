import types
from kipr_array import *
from . import unicode_simplify
from . import sys_utils as sys

text = types.SimpleNamespace()
text.normalization_map = unicode_simplify.normalize