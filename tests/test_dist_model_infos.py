#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-12 21:47:09
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-13 09:37:36
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
@click.option('--model-infos-filename', required=True, type=str, help='Model Infos Filename.')
@click.option('--load-dirpath',         required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Model Infos load directory.')
@click.option('--save-dirpath',         required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Model Infos save directory.')
@click.option('--framework',            required=True, type=click.Choice(['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], case_sensitive=True), help='Indicates the framework to which the model belonged prior to conversion.')
@click.option('--limit-l',              required=True, type=int, help='Indicates the range left border model size limit.')
@click.option('--limit-r',              required=True, type=int, help='Indicates the range right border model size limit.')
@click.option('--number-of-file',       required=True, type=int, help='Indicates the number of files to save.')
def main(model_infos_filename: str, load_dirpath: pathlib.Path, save_dirpath: pathlib.Path, framework: str, limit_l: int, limit_r: int, number_of_file: int):
    load_model_infos_filename = f'{model_infos_filename}'
    save_model_infos_filename = f'{model_infos_filename}_{limit_l}_{limit_r}'
    gb = 1024*1024*1024
    l_model_size_limit = (0, limit_l*gb)
    r_model_size_limit = (0, limit_r*gb)
    l_model_infos = list()
    r_model_infos = list()

    index = 0
    load_model_infos_filepath = load_dirpath.joinpath(f'{load_model_infos_filename}_{index}.json')
    while load_model_infos_filepath.is_file():
        logger.info(f' -> {index}. Model Infos Load From Dir: \'{load_dirpath}\'.')
        index += 1
        l_load_model_infos, _ = get_model_infos_and_convert_method(load_model_infos_filepath, framework, l_model_size_limit)
        r_load_model_infos, _ = get_model_infos_and_convert_method(load_model_infos_filepath, framework, r_model_size_limit)
        l_model_infos.extend(l_load_model_infos)
        r_model_infos.extend(r_load_model_infos)

        load_model_infos_filepath = load_dirpath.joinpath(f'{load_model_infos_filename}_{index}.json')

    model_ids = set()
    if l_model_size_limit[1] != 0:
        for l_model_info in l_model_infos:
            model_ids.add(l_model_info['id'])

    model_infos = list()
    for r_model_info in r_model_infos:
        if r_model_info['id'] not in model_ids:
            model_infos.append(r_model_info)

    quotient, remainder = divmod(len(model_infos), number_of_file)
    l_index = 0
    r_index = 0
    for index in range(number_of_file):
        logger.info(f' -> {index}. Model Infos Save Into Dir: \'{load_dirpath}\'.')
        r_index = l_index + quotient + (1 if index < remainder else 0)
        save_model_infos = model_infos[l_index:r_index]
        save_json(save_model_infos, save_dirpath.joinpath(f'{save_model_infos_filename}_{index}.json'))
        l_index = r_index
        logger.info(f'Model Infos Save Into Dir: \'{save_dirpath}\', total={len(save_model_infos)}.')

    logger.info(f'Done.')


if __name__ == '__main__':
    main()
