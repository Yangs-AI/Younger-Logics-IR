#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-13 14:25:10
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-26 03:50:48
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import networkx

from .io import check_model
from .translation import trans_model_proto


def convert_onnx_to_ir(model_handler: onnx.ModelProto) -> networkx.DiGraph:
    """
    .. note::
        Do not check the model_handler now, cause a model without external data file can also be loaded.
        The error should be catched by the callers.
    """

    assert isinstance(model_handler, onnx.ModelProto), f'Argument \"model_handler\" must be an ONNX Model Proto (onnx.ModelProto) instead \"{type(model_handler)}\"!'
    # if check_model(model_handler):
    #     dag = trans_model_proto(model_handler, neglect_tensor_values=True)
    # else:
    #     """
    #     .. todo::
    #         If Check Is Invalid, There Is No model
    #     """
    #     dag = None
    dag = trans_model_proto(model_handler, neglect_tensor_values=True)
    return dag
