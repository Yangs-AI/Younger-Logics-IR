#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:56:56
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import pathlib

from typing import Literal

from younger.commons.io import save_json

from younger_logics_ir.commons.logging import logger

from .utils import get_huggingface_hub_model_infos, get_huggingface_hub_model_ids, get_huggingface_hub_metric_infos, get_huggingface_hub_metric_ids, get_huggingface_hub_task_infos, get_huggingface_hub_task_ids


def save_huggingface_model_infos(save_dirpath: pathlib.Path, token: str | None = None, number_per_file: int | None = None, worker_number: int | None = None):
    times = 0
    finished = False
    while not finished:
        try:
            get_huggingface_hub_model_infos(save_dirpath, token=token, number_per_file=number_per_file, worker_number=worker_number)
            finished = True
        except:
            finished = False
            times += 1
            logger.info(f'!!! Not Finished (No. 1): Another Try ... ')


def save_huggingface_model_ids(save_dirpath: pathlib.Path, token: str | None = None, number_per_file: int | None = None) -> None:
    model_ids_per_file = list()
    svd_model_id_index = 0
    cur_model_id_index = 0
    for model_id in get_huggingface_hub_model_ids(token=token):
        model_ids_per_file.append(model_id)
        if number_per_file is not None and len(model_ids_per_file) == number_per_file:
            save_filepath = save_dirpath.joinpath(f'huggingface_model_ids_{svd_model_id_index}_{cur_model_id_index}.json')
            save_json(model_ids_per_file, save_filepath, indent=2)
            logger.info(f'Total {len(model_ids_per_file)} Model ID Items Saved In: \'{save_filepath}\'.')
            svd_model_id_index = cur_model_id_index
        cur_model_id_index += 1

    if number_per_file is not None and len(model_ids_per_file) == number_per_file:
        save_filepath = save_dirpath.joinpath(f'huggingface_model_ids_{svd_model_id_index}_{cur_model_id_index}.json')
        save_json(model_ids_per_file, save_filepath, indent=2)
        logger.info(f'Total {len(model_ids_per_file)} Model ID Items Saved In: \'{save_filepath}\'.')
    logger.info(f'Finished. Total {cur_model_id_index} Model IDs.')


def save_huggingface_metric_infos(save_dirpath: pathlib.Path, token: str | None = None) -> None:
    times = 0
    finished = False
    while not finished:
        try:
            get_huggingface_hub_metric_infos(save_dirpath, token=token)
            finished = True
        except:
            finished = False
            times += 1
            logger.info(f'!!! Not Finished (No. 1): Another Try ... ')


def save_huggingface_metric_ids(save_dirpath: pathlib.Path, token: str | None = None) -> None:
    metric_ids = get_huggingface_hub_metric_ids(token=token)
    save_filepath = save_dirpath.joinpath('huggingface_metric_ids.json')
    save_json(metric_ids, save_filepath, indent=2)
    logger.info(f'Total {len(metric_ids)} Metric IDs. Results Saved In: \'{save_filepath}\'.')


def save_huggingface_task_infos(save_dirpath: pathlib.Path, token: str | None = None) -> None:
    times = 0
    finished = False
    while not finished:
        try:
            get_huggingface_hub_task_infos(save_dirpath, token=token)
            finished = True
        except:
            finished = False
            times += 1
            logger.info(f'!!! Not Finished (No. 1): Another Try ... ')


def save_huggingface_task_ids(save_dirpath: pathlib.Path, token: str | None = None) -> None:
    task_ids = get_huggingface_hub_task_ids(token=token)
    save_filepath = save_dirpath.joinpath('huggingface_task_ids.json')
    save_json(task_ids, save_filepath, indent=2)
    logger.info(f'Total {len(task_ids)} Task IDs. Results Saved In: \'{save_filepath}\'.')


def main(mode: Literal['Model_Infos', 'Model_IDs', 'Metric_Infos', 'Metric_IDs', 'Task_Infos', 'Task_IDs'], save_dirpath: pathlib.Path, mirror_url: str, **kwargs) -> None:
    assert mode in {'Model_Infos', 'Model_IDs', 'Metric_Infos', 'Metric_IDs', 'Task_Infos', 'Task_IDs'}

    os.environ['HF_ENDPOINT'] = 'https://huggingface.co/' if mirror_url == '' else mirror_url

    if mode == 'Model_Infos':
        save_huggingface_model_infos(save_dirpath, token=kwargs['token'], number_per_file=kwargs['number_per_file'], worker_number=kwargs['worker_number'])
        return

    if mode == 'Model_IDs':
        save_huggingface_model_ids(save_dirpath, token=kwargs['token'], number_per_file=kwargs['number_per_file'])
        return

    if mode == 'Metric_Infos':
        save_huggingface_metric_infos(save_dirpath, token=kwargs['token'])
        return

    if mode == 'Metric_IDs':
        save_huggingface_metric_ids(save_dirpath, token=kwargs['token'])
        return

    if mode == 'Task_Infos':
        save_huggingface_task_infos(save_dirpath, token=kwargs['token'])
        return

    if mode == 'Task_IDs':
        save_huggingface_task_ids(save_dirpath, token=kwargs['token'])
        return
