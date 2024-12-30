#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-30 09:26:11
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import onnx
import tqdm
import torch
import pathlib
import multiprocessing

from typing import Any, Literal

from younger.commons.io import loads_json, saves_json, create_dir, delete_dir, load_json
from younger.commons.logging import logger

from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert
from younger_logics_ir.commons.constants import YLIROriginHub
from younger_logics_ir.scripts.commons.utils import get_onnx_opset_versions

from .utils import get_torch_hub_model_input, get_torch_hub_model_model_instance


def safe_torch_export(cvt_cache_dirpath: pathlib.Path, model_id: str, results_queue: multiprocessing.Queue):
    this_status: dict[int, str] = dict()
    this_instances: list[Instance] = list()
    model = get_torch_hub_model_model_instance(model_id)
    model_input = get_torch_hub_model_input(model_id)
    for onnx_opset_version in get_onnx_opset_versions():
        onnx_model_path = str(cvt_cache_dirpath.joinpath(f'model_opset_v{onnx_opset_version}.onnx'))
        try:
            torch.onnx.export(model, model_input, onnx_model_path, verbose=True)
            onnx_model = onnx.load(onnx_model_path)
            try:
                instance = Instance()
                instance.setup_logicx(convert(onnx_model))
                this_instances.append(instance)
                this_status[onnx_opset_version] = 'success'
            except Exception as exception:
                this_status[onnx_opset_version] = 'onnx2logicx_convert_error'
        except Exception as exception:
            this_status[onnx_opset_version] = 'torch2onnx_convert_error'
        delete_dir(cvt_cache_dirpath, only_clean=True)

    results_queue.put((this_status, this_instances))


def convert_torch(model_id: str, cvt_cache_dirpath: pathlib.Path) -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()

    results_queue = multiprocessing.Queue()
    subprocess = multiprocessing.Process(target=safe_torch_export, args=(cvt_cache_dirpath, model_id, results_queue))
    subprocess.start()
    subprocess.join()
    if results_queue.empty():
        this_status = 'system_kill'
        this_instances = list()
    else:
        # this_status: dict[int, str]
        # this_instances: list[Instance]
        this_status, this_instances = results_queue.get()

    status['main_model'] = this_status
    instances.extend(this_instances)

    return status, instances


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    status_filepath = sts_cache_dirpath.joinpath(f'default.sts')
    if status_filepath.is_file():
        with open(status_filepath, 'r') as status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f'default_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f'default.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f'default_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


def main(
    model_infos_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
):
    model_infos: list[dict[str, Any]] = load_json(model_infos_filepath)

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # Official
    # ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-OXOfc')
    # create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-TRCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-TRSts')
    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath)
    number_of_converted_models = len(convert_status)
    logger.info(f'-> Previous Converted Models: {number_of_converted_models}')

    logger.info(f'-> Instances Creating ...')
    with tqdm.tqdm(total=len(model_infos), desc='Create Instances') as progress_bar:
        for convert_index, model_info in enumerate(model_infos, start=1):
            model_id = model_info['id']

            if last_handled_model_id is not None:
                if model_id == last_handled_model_id:
                    last_handled_model_id = None
                progress_bar.set_description(f'Converted, Skip - {model_id}')
                progress_bar.update(1)
                continue

            model_input = get_torch_hub_model_input(model_id)
            if model_input is None:
                set_convert_status_last_handled_model_id(sts_cache_dirpath, dict(before_convert='unknown_input'), model_id)
                progress_bar.set_description(f'[E](unknow_input) - {model_id}')
                progress_bar.update(1)
                continue

            status, instances = convert_torch(model_id, cvt_cache_dirpath)

            set_convert_status_last_handled_model_id(sts_cache_dirpath, status, model_id)
            # delete_dir(ofc_cache_dirpath, only_clean=True)

            model_owner, model_name = model_id.split('/')
            for instance_index, instance in enumerate(instances, start=1):
                instance.insert_label(
                    Implementation(
                        origin=Origin(YLIROriginHub.ONNX, model_owner, model_name),
                        like=model_info['likes'],
                        download=model_info['downloadsAllTime'],
                    )
                )
                instance.save(instances_dirpath.joinpath(instance.unique))

            progress_bar.set_description(f'Convert - {model_id}')
            progress_bar.update(1)

    logger.info(f'-> Instances Created.')
