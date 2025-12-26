#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:48:46
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import sys
import onnx
import pathlib

from younger.commons.io import create_dir

from younger_logics_ir.commons.logging import logger


def check_model(model_handler: onnx.ModelProto | pathlib.Path) -> bool:
    assert isinstance(model_handler, onnx.ModelProto) or isinstance(model_handler, pathlib.Path)
    # Change Due To Hash May Lead OOM.
    def check_with_internal() -> str | None:
        model = model_handler
        if len(model.graph.node) == 0:
            check_result = False
        else:
            onnx.checker.check_model(model)
            #check_result = hash_bytes(model)
            check_result = True
        return check_result

    def check_with_external() -> str | None:
        onnx.checker.check_model(str(model_handler))
        #model = onnx.load(str(model_handler))
        #check_result = hash_bytes(model.SerializeToString())
        check_result = True

        return check_result

    try:
        if isinstance(model_handler, onnx.ModelProto):
            return check_with_internal()
        if isinstance(model_handler, pathlib.Path):
            return check_with_external()
    except onnx.checker.ValidationError as check_error:
        """
        Missing of external data file also can result in this error.
        """
        logger.warning(f'The ONNX Model is invalid: {check_error}')
        check_result = False
    except Exception as exception:
        logger.error(f'An error occurred while checking the ONNX model: {exception}')
        sys.exit(1)
    return check_result


def load_model(model_filepath: pathlib.Path) -> onnx.ModelProto:
    model = onnx.load(model_filepath, load_external_data=False)
    return model


def save_model(model: onnx.ModelProto, model_filepath: pathlib.Path) -> None:
    create_dir(model_filepath.parent)
    onnx.save(model, model_filepath)
    return
