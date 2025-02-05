#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-05 15:26:30
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import onnx
import tqdm
import pathlib
import multiprocessing

from typing import Any, Literal, Callable

from huggingface_hub import login, hf_hub_download, snapshot_download, utils

from younger.commons.io import saves_json, loads_json, save_json, load_json, create_dir, delete_dir, get_human_readable_size_representation
from younger.commons.hash import hash_string
from younger.commons.logging import logger

from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert
from younger_logics_ir.converters.onnx2ir.io import load_model

from younger_logics_ir.commons.constants import YLIROriginHub

from younger_logics_ir.scripts.commons.utils import get_onnx_opset_versions, get_onnx_model_opset_version

from .utils import get_huggingface_hub_model_readme, get_huggingface_hub_model_siblings, clean_huggingface_hub_model_cache, infer_supported_frameworks


def clean_cache(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path):
    delete_dir(cvt_cache_dirpath, only_clean=True)
    clean_huggingface_hub_model_cache(model_id, ofc_cache_dirpath)


def safe_optimum_export(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, device: str):
    from optimum.exporters.onnx import main_export

    try:
        main_export(model_id, cvt_cache_dirpath, opset=onnx_opset_version, device=device, cache_dir=ofc_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
        this_status = 'success'
    except MemoryError as exception:
        this_status = 'oversize'
    except utils.RepositoryNotFoundError as exception:
        this_status = 'access_deny'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_optimum(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]], list[Instance]]:
    assert device in {'cpu', 'cuda'}
    status: dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]] = dict()
    instances: list[Instance] = list()

    for onnx_opset_version in get_onnx_opset_versions():
        # The highest opset version supported by torch.onnx.export is 20
        if onnx_opset_version > 20:
            continue
        results_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_optimum_export, args=(model_id, cvt_cache_dirpath, ofc_cache_dirpath, onnx_opset_version, results_queue, device))
        subprocess.start()
        subprocess.join()

        this_status_details: dict[str, Literal['success', 'logicx_error']] = dict()
        if results_queue.empty():
            this_status = 'system_kill'
        else:
            this_status = results_queue.get()
            if this_status == 'success':
                for filepath in cvt_cache_dirpath.rglob('*.onnx'):
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(filepath)))
                        instances.append(instance)
                        this_status_details[str(filepath)] = 'success'
                    except Exception as exception:
                        this_status_details[str(filepath)] = 'logicx_error'
        status[onnx_opset_version] = (this_status, this_status_details)

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_keras_export(keras_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export

    if keras_model_path.is_dir():
        model_type = 'saved_model'
    if keras_model_path.is_file():
        model_type = 'keras'

    try:
        tf2onnx_main_export(keras_model_path, onnx_model_path, onnx_opset_version, model_type=model_type)
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_keras(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    remote_keras_model_paths = list()
    for remote_keras_model_path in get_huggingface_hub_model_siblings(model_id, suffixes=['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']):
        if remote_keras_model_path.endswith('.pbtxt') or remote_keras_model_path.endswith('.pb'):
            remote_keras_model_paths.append(os.path.dirname(remote_keras_model_path))
        else:
            remote_keras_model_paths.append(remote_keras_model_path)
    remote_keras_model_paths = list(set(remote_keras_model_paths))

    for remote_keras_model_path in remote_keras_model_paths:
        remote_keras_model_name = os.path.splitext(remote_keras_model_path)[0]
        try:
            if os.path.isdir(remote_keras_model_path):
                keras_model_path = pathlib.Path(snapshot_download(model_id, allow_patterns=f'{remote_keras_model_path}/*', cache_dir=ofc_cache_dirpath)).joinpath(remote_keras_model_path)
            else:
                keras_model_path = pathlib.Path(hf_hub_download(model_id, remote_keras_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_keras_model_name] = 'access_deny'
            continue

        status[remote_keras_model_name] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(keras_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_keras_export, args=(keras_model_path, onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()

            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                    except Exception as exception:
                        this_status = 'logicx_error'
            status[remote_keras_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_tflite_export(tflite_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export

    try:
        tf2onnx_main_export(tflite_model_path, onnx_model_path, onnx_opset_version, model_type='tflite')
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_tflite(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    remote_tflite_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.tflite'])
    for remote_tflite_model_path in remote_tflite_model_paths:
        remote_tflite_model_name = os.path.splitext(remote_tflite_model_path)[0]
        try:
            tflite_model_path = pathlib.Path(hf_hub_download(model_id, remote_tflite_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_tflite_model_name] = 'access_deny'
            continue

        status[remote_tflite_model_name] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(tflite_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_tflite_export, args=(tflite_model_path, onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()
            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                    except Exception as exception:
                        this_status = 'logicx_error'

            status[remote_tflite_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_onnx_export(origin_version_onnx_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version, results_queue: multiprocessing.Queue):
    try:
        origin_version_onnx_model = load_model(origin_version_onnx_model_path)
        # Convert the ONNX model to the target opset version. This step will not occupy disk space. Thus cvt_cache_dirpath is not used.
        onnx.save_model(onnx.version_converter.convert_version(origin_version_onnx_model, onnx_opset_version), onnx_model_path)
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_onnx(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    remote_onnx_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.onnx'])
    for remote_onnx_model_path in remote_onnx_model_paths:
        remote_onnx_model_name = os.path.splitext(remote_onnx_model_path)[0]
        onnx_model_path = pathlib.Path(hf_hub_download(model_id, remote_onnx_model_path, cache_dir=ofc_cache_dirpath))

        status[remote_onnx_model_name] = dict()
        onnx_model = load_model(onnx_model_path)
        onnx_model_opset_version = get_onnx_model_opset_version(onnx_model)
        for onnx_opset_version in get_onnx_opset_versions():
            if onnx_opset_version == onnx_model_opset_version:
                continue
            other_version_onnx_model_path = cvt_cache_dirpath.joinpath(f'{str(onnx_model_path.with_suffix(""))}-{onnx_opset_version}.onnx')
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_onnx_export, args=(onnx_model_path, other_version_onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()
            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(other_version_onnx_model_path)))
                        instances.append(instance)
                    except Exception as exception:
                        this_status = 'logicx_error'
            status[remote_onnx_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def get_model_infos_and_convert_method(model_infos_filepath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite'], model_size_limit: tuple[int, int]) -> tuple[list[dict[str, Any]], Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]]]:
    # This is the convert order.
    supported_frameworks: list[str] = ['optimum', 'keras', 'onnx', 'tflite']
    supported_convert_methods: dict[str, Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]]] = dict(
        optimum=convert_optimum,
        onnx=convert_onnx,
        keras=convert_keras,
        tflite=convert_tflite,
    )

    def get_model_frameworks(model_frameworks: list[Literal['optimum', 'onnx', 'keras', 'tflite']]) -> Literal['optimum', 'onnx', 'keras', 'tflite']:
        candidate_frameworks = set(model_frameworks) & set(supported_frameworks)
        model_frameworks = list()
        for supported_framework in supported_frameworks:
            if supported_framework in candidate_frameworks:
                model_frameworks.append(supported_framework)
        return model_frameworks

    model_infos: list[dict[str, Any]] = list()
    for model_info in load_json(model_infos_filepath):
        model_frameworks = get_model_frameworks(infer_supported_frameworks(model_info))
        if framework in model_frameworks and model_size_limit[0] <= model_info['usedStorage'] and model_info['usedStorage'] <= model_size_limit[1]:
            model_infos.append(model_info)

    convert_method = supported_convert_methods[framework]
    return model_infos, convert_method


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite'], model_size_limit: tuple[int, int]) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    specific_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    if specific_status_filepath.is_file():
        with open(specific_status_filepath, 'r') as specific_status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in specific_status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite'], model_size_limit: tuple[int, int], convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


def main(
    model_infos_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
    device: Literal['cpu', 'cuda'] = 'cpu',
    framework: Literal['optimum', 'onnx', 'keras', 'tflite'] = 'optimum',
    model_size_limit_l: int | None = None,
    model_size_limit_r: int | None = None,
    token: str | None = None,
    estimate: bool = False,
):
    """
    Retrieve Metadata of HuggingFace Models and Save Them Into Files.

    :param model_ids_filepath: _description_
    :type model_ids_filepath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    :param cache_dirpath: _description_
    :type cache_dirpath: pathlib.Path
    :param device: _description_, defaults to 'cpu'
    :type device: Literal[&#39;cpu&#39;, &#39;cuda&#39;], optional
    :param framework: _description_, defaults to 'optimum'
    :type framework: Literal[&#39;optimum&#39;, &#39;onnx&#39;, &#39;keras&#39;, &#39;tflite&#39;], optional
    :param model_size_limit_l: _description_, defaults to None
    :type model_size_limit_l: int | None, optional
    :param model_size_limit_r: _description_, defaults to None
    :type model_size_limit_r: int | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    In this project we have a concept called Origin. Origin is a tuple of (hub, owner, name).

    HuggingFace Hub is a place where people can share their models, datasets, and scripts.
    This project hardcodes the hub as 'HuggingFace'.
    The Naming Convention of the Mdoel ID on HuggingFace Hub follows the format: {owner}/{name}.
    Thus the Origin of a Implementation, often called as Model which is a instance of a LogicX a.k.a. Neural Network Architecture (NNA), on HuggingFace Hub is Origin('HuggingFace', owner, name).

    Model Infos are retrieved from the HuggingFace Hub and sorted with lastModified time in descending order, and saved into a JSON file by using command `younger-logics-ir create onnx retrieve huggingface --mode Model_Infos --save-dirpath ${SAVE_DIRPATH}`.

    .. note::
        The Instances are saved into the directory named as 'Instances-HuggingFace-{Framework}' under the save_dirpath.

    """

    model_size_limit_l = model_size_limit_l or 0
    logger.info(f'   Model Size Left Limit: {get_human_readable_size_representation(model_size_limit_l)}.')

    model_size_limit_r = model_size_limit_r or 1024 * 1024 * 1024 * 1024 * 1024
    logger.info(f'   Model Size Right Limit: {get_human_readable_size_representation(model_size_limit_r)}.')

    model_size_limit = (model_size_limit_l, model_size_limit_r)

    model_infos, convert_method = get_model_infos_and_convert_method(model_infos_filepath, framework, model_size_limit)
    if estimate:
        logger.info(f'Only Estimate. Models To Be Converted: {len(model_infos)}; Model Infos Filename: {model_infos_filepath.name}.')
        return

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # READMES
    readmes_dirpath = save_dirpath.joinpath(f'READMES')
    create_dir(instances_dirpath)

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFOfc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFSts')
    create_dir(sts_cache_dirpath)
    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit)
    number_of_converted_models = len(convert_status)
    logger.info(f'-> Previous Converted Models: {number_of_converted_models}')

    if token is not None:
        logger.info(f'-> HuggingFace Token Provided. Now Logging In ...')
        login(token)
    else:
        logger.info(f'-> HuggingFace Token Not Provided. Now Accessing Without Token ...')

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

            status, instances = convert_method(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device)

            model_owner, model_name = model_id.split('/')
            for instance_index, instance in enumerate(instances, start=1):
                instance.insert_label(
                    Implementation(
                        origin=Origin(YLIROriginHub.HUGGINGFACE, model_owner, model_name),
                        like=model_info['likes'],
                        download=model_info['downloadsAllTime'],
                    )
                )
                instance.save(instances_dirpath.joinpath(instance.unique))

            try:
                readme = get_huggingface_hub_model_readme(model_id, token=token)
            except Exception as exception:
                readme = ''

            if readme == '':
                pass
            else:
                save_json(readme, readmes_dirpath.joinpath(f'{model_owner}_YLIR_{model_name}.json'))

            set_convert_status_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit, status, model_id)
            clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)

            progress_bar.set_description(f'Convert - {model_id}')
            progress_bar.update(1)

    logger.info(f'-> Instances Created.')
