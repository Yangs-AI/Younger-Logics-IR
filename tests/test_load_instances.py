#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-03 15:44:26
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-26 03:52:49
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import click
import pathlib


from younger.commons.io import save_json
from younger_logics_ir.modules import Instance, LogicX, Implementation, Origin
from younger_logics_ir.converters import convert

from younger_logics_ir.commons.constants import YLIROriginHub
import json


def find_bytes_in_dict(d):
    for key, value in d.items():
        if isinstance(value, bytes):
            print(f"Found bytes at key: {key}, value: {value}")
        elif isinstance(value, dict):
            find_bytes_in_dict(value)
        elif isinstance(value, list):
            find_bytes_in_list(value)

def find_bytes_in_list(lst):
    for i, item in enumerate(lst):
        if isinstance(item, bytes):
            print(f"Found bytes at index {i}: {item}")
        elif isinstance(item, dict):
            find_bytes_in_dict(item)
        elif isinstance(item, list):
            find_bytes_in_list(item)


@click.command()
@click.option('--load-dirpath', type=pathlib.Path, required=True)
def main(load_dirpath: pathlib.Path):

    for index, instance_dirpath in enumerate(load_dirpath.iterdir()):
        instance = Instance()

        # Convert the ONNX model to YLIR instance
        instance.load(instance_dirpath)
        print(instance.unique)
        # sd = LogicX.saved_dag(instance.logicx.dag)

        # find_bytes_in_dict(sd)


if __name__ == '__main__':
    main()
