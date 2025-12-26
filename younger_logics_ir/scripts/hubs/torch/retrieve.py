#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:58:16
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from typing import Literal

from younger.commons.io import load_json, save_json

from younger_logics_ir.commons.logging import logger

from .utils import get_torch_hub_model_infos, get_torch_hub_model_ids


def save_torch_hub_model_infos(save_dirpath: pathlib.Path, force_reload: bool | None = None) -> None:
    save_filepath = save_dirpath.joinpath(f'torch_model_infos.json')
    if save_filepath.is_file() and not force_reload:
        model_infos = load_json(save_filepath)
        logger.info(f' -> Already Retrieved. Total {len(model_infos)} Model Infos. Results From: \'{save_filepath}\'.')
    else:
        model_infos = get_torch_hub_model_infos()
        logger.info(f' -> Total {len(model_infos)} Model Infos.')
        logger.info(f' v Saving Results Into {save_filepath} ...')
        save_json(model_infos, save_filepath, indent=2)
        logger.info(f' ^ Saved.')
    logger.info(f' => Finished')


def save_torch_hub_model_ids(save_dirpath: pathlib.Path):
    model_ids = get_torch_hub_model_ids()
    save_filepath = save_dirpath.joinpath(f'torch_model_ids.json')
    save_json(model_ids, save_filepath, indent=2)
    logger.info(f'Total {len(model_ids)} Model IDs. Results Saved In: \'{save_filepath}\'.')


def main(mode: Literal['Model_Infos', 'Model_IDs'], save_dirpath: pathlib.Path, **kwargs):
    assert mode in {'Model_Infos', 'Model_IDs'}

    if mode == 'Model_Infos':
        save_torch_hub_model_infos(save_dirpath, force_reload=kwargs['force_reload'])
        return

    if mode == 'Model_IDs':
        save_torch_hub_model_ids(save_dirpath)
        return
