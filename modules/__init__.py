from __future__ import absolute_import

import sys
import importlib
#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
#globals().update({name: module_dict[name] for name in to_import})
sys.path.append(mdl.homedir+'/modules')


from .useful import *
from .define_H import *
from .define_Hm import *
from .define_Hw import *
from .check_budget import *
from .define_error import *
from .interpreter import *
from .LRU_map import *
from .posterior_map import *
from .AnalyticInv import *
from .misfit import *
from .budget_lat import *
from .diagnostics import *
from .diagnostics_map import *
from .redim_H import *
from .extract_SIF import *
from .consistency import *
from .launch_lmdz import *

#from .footprint import *


