#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-05 15:30:32
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import importlib.metadata

from younger.commons.constants import YoungerHandle


__version__ = importlib.metadata.version("younger_logics_ir")


__thename__ = YoungerHandle.LogicsName + '-' + 'IR'


# Import submodules after __thename__ is defined to avoid circular import
from . import commons
from . import modules
from . import converters
