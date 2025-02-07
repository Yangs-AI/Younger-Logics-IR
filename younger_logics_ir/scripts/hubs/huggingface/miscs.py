#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-05 16:24:43
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib
import tensorflow

from typing import Literal

from tf2onnx import tf_loader, optimizer, utils
from tf2onnx.graph import ExternalTensorStorage
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.tf_utils import compress_graph_def

from younger.commons.io import get_path_size
from younger.commons.logging import logger

assert tensorflow.__version__ == '2.15.0', 'The current version of tensorflow is not supported by the current version of tf2onnx.'


def tf2onnx_main_export(model_path: pathlib.Path, output_path: pathlib.Path, opset: int, model_type: Literal['saved_model', 'keras', 'tflite', 'tfjs'] = 'saved_model', logging: bool = True, directly_return: bool = False):
    # [NOTE] The Code are modified based on the official tensorflow-onnx source codes. (https://github.com/onnx/tensorflow-onnx/blob/main/tf2onnx/convert.py [Method: main])
    assert model_type in {'saved_model', 'keras', 'tflite', 'tfjs'}

    out_logger_flag = logger.disabled
    logger.disabled = not logging

    model_name = model_path.name
    large_model = 2*1024*1024*1024 < get_path_size(model_path) 
    tfjs_filepath = None
    tflite_filepath = None
    frozen_graph = None
    inputs = None
    outputs = None
    external_tensor_storage = None
    const_node_values = None

    if model_type == 'saved_model':
        frozen_graph, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(
            model_path, inputs, outputs, return_initialized_tables=True, return_tensors_to_rename=True
        )

    if model_type == 'keras':
        frozen_graph, inputs, outputs = tf_loader.from_keras(
            model_path, inputs, outputs
        )

    if model_type == 'tflite':
        tflite_filepath = model_path

    if model_type == 'tfjs':
        tfjs_filepath = model_path

    with tensorflow.device("/cpu:0"):
        with tensorflow.Graph().as_default() as tensorflow_graph:
            if large_model:
                const_node_values = compress_graph_def(frozen_graph)
                external_tensor_storage = ExternalTensorStorage()
            if model_type not in {'tflite', 'tfjs'}:
                tensorflow.import_graph_def(frozen_graph, name='')
            graph = process_tf_graph(
                tensorflow_graph,
                const_node_values=const_node_values,
                input_names=inputs,
                output_names=outputs,
                tflite_path=tflite_filepath,
                tfjs_path=tfjs_filepath,
                opset=opset
            )
            onnx_graph = optimizer.optimize_graph(graph, catch_errors=True)
            model_proto = onnx_graph.make_model(f'converted from {model_name}', external_tensor_storage=external_tensor_storage)

    logger.info(f'   ... Successfully converted TensorFlow model {model_path} to ONNX')

    if directly_return:
        return model_proto
    else:
        if large_model:
            utils.save_onnx_zip(output_path, model_proto, external_tensor_storage)
            logger.info(f'   ... Zipped ONNX model is saved at {output_path}. Unzip before opening in onnxruntime.')
        else:
            utils.save_protobuf(output_path, model_proto)
            logger.info(f'   ... ONNX model is saved at {output_path}')

    logger.disabled = out_logger_flag
