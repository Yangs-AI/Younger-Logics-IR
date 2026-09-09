#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-11 13:53:22
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-13 10:53:58
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger_logics_ir.scripts.hubs.huggingface.convert import get_model_infos_and_convert_method


@click.command()
@click.option('--model-infos-dirpath',  required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Model Infos directory.')
@click.option('--save-filepath',        required=True,  type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='Count Saving directory.')
@click.option('--model-infos-filename', required=True,  type=str, help='Model Infos Filename.')
@click.option('--framework',            required=False, type=click.Choice(['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], case_sensitive=True), default='optimum', help='Indicates the framework to which the model belonged prior to conversion.')
def main(model_infos_dirpath: pathlib.Path, save_filepath: pathlib.Path, model_infos_filename: str, framework: str):
    gb = 1024*1024*1024
    model_size_limit_rs = [0.5, 1, 2, 4, 8, 12, 16, 32, 64, 1000, 1000*1000, 1000*1000*1000]
    cum_counts = dict()
    for model_size_limit_r in model_size_limit_rs:
        cum_counts[model_size_limit_r] = 0

    index = 0
    model_infos_filepath = model_infos_dirpath.joinpath(f'{model_infos_filename}_{index}.json')
    while model_infos_filepath.is_file():
        index += 1
        for model_size_limit_r in model_size_limit_rs:
            model_infos, _ = get_model_infos_and_convert_method(model_infos_filepath, framework, (0, model_size_limit_r*gb))
            cum_counts[model_size_limit_r] += len(model_infos)

        logger.info(f'Checked Model Infos File: \'{model_infos_filepath.name}\'.')
        model_infos_filepath = model_infos_dirpath.joinpath(f'{model_infos_filename}_{index}.json')

    itv_counts = dict()
    for index in range(len(model_size_limit_rs)):
        r = model_size_limit_rs[index]
        if index == 0:
            itv_counts[r] = cum_counts[r]
        else:
            itv_counts[r] = cum_counts[r] - cum_counts[model_size_limit_rs[index-1]]
    counts = dict()
    for model_size_limit_r in model_size_limit_rs:
        counts[model_size_limit_r] = (cum_counts[model_size_limit_r], itv_counts[model_size_limit_r])

    save_json(counts, save_filepath, indent=2)
    logger.info(f'Counts Saved Into: \'{save_filepath}\'.')


if __name__ == '__main__':
    main()
