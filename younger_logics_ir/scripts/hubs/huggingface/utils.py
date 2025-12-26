#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:57:12
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import tqdm
import pathlib
import requests
import multiprocessing.pool

from typing import Any, Literal, Generator
from huggingface_hub import utils, HfFileSystem, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from younger.commons.io import save_json, delete_dir, get_human_readable_size_representation
from younger.commons.cache import CachedChunks
from younger.commons.string import extract_possible_digits_from_readme_string, extract_possible_tables_from_readme_string, split_front_matter_from_readme_string, README_DATE_Pattern, README_DATETIME_Pattern, README_TABLE_Pattern

from younger_logics_ir.commons.cache import YLIR_CACHE_ROOT
from younger_logics_ir.commons.logging import logger


HUGGINGFACE_HUB_API_ENDPOINT = 'https://huggingface.co/api'


def get_one_data_from_huggingface_hub_api(path: str, params: dict | None = None, token: str | None = None) -> Any:
    """
    _summary_

    :param path: _description_
    :type path: str
    :param params: _description_, defaults to None
    :type params: dict | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    :yield: _description_
    :rtype: Any

    .. note::
        The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_pagination.py)
        `paginate` is called by list_models, list_datasets, and list_spaces methods of huggingface_hub.HfApi;
    """

    params = params or dict()

    session = requests.Session()
    headers = utils.build_hf_headers(token=token)
    response = session.get(path, params=params, headers=headers)
    try:
        utils.hf_raise_for_status(response)
        return response.json()
    except utils.HfHubHTTPError as e:
        print(f'Request: {path} {e.request_id} - {e.server_message}')
        return None


def get_all_data_from_huggingface_hub_api(path: str, params: dict | None = None, token: str | None = None) -> Generator[Any, None, None]:
    """
    _summary_

    :param path: _description_
    :type path: str
    :param params: _description_, defaults to None
    :type params: dict | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    :yield: _description_
    :rtype: Generator[Any, None, None]

    .. note::
        The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_pagination.py)
        `paginate` is called by list_models, list_datasets, and list_spaces methods of huggingface_hub.HfApi;
    """

    params = params or dict()

    session = requests.Session()
    headers = utils.build_hf_headers(token=token)
    response = session.get(path, params=params, headers=headers)
    try:
        utils.hf_raise_for_status(response)
        yield from response.json()
        next_page_path = response.links.get("next", {}).get("url")
    except utils.HfHubHTTPError as e:
        print(f'Request: {path} {e.request_id} - {e.server_message}')
        yield from ()
        next_page_path = None

    # Follow pages
    # Next link already contains query params
    while next_page_path is not None:
        logger.debug(f"Pagination detected. Requesting next page: {next_page_path}")
        response = session.get(next_page_path, headers=headers)
        try:
            utils.hf_raise_for_status(response)
            yield from response.json()
            next_page_path = response.links.get("next", {}).get("url")
        except utils.HfHubHTTPError as e:
            print(f'Request: {path} {e.request_id} - {e.server_message}')
            yield from ()
            next_page_path = None


#############################################################################################################
# vvv Below is the functions to get the information of models, metrics, and tasks from Hugging Face Hub vvv #
#############################################################################################################


def get_huggingface_hub_model_ids(token: str | None = None) -> Generator[str, None, None]:
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'
    models = get_all_data_from_huggingface_hub_api(models_path, params=dict(sort='lastModified', direction=-1, expand=['lastModified']), token=token)
    return (model['id'] for model in models)


def get_huggingface_hub_metric_ids(token: str | None = None) -> list[str]:
    metrics_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/metrics'
    metrics = get_all_data_from_huggingface_hub_api(metrics_path, token=token)
    metric_ids = [metric['id'] for metric in metrics]
    return metric_ids


def get_huggingface_hub_task_ids(token: str | None = None) -> list[str]:
    tasks_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/tasks'
    tasks = get_all_data_from_huggingface_hub_api(tasks_path, token=token)
    task_ids = [task_id for task_id, task_info in tasks.items()]
    return task_ids


def get_huggingface_hub_model_infos(save_dirpath: pathlib.Path, token: str | None = None, number_per_file: int | None = None, worker_number: int | None = None):
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'

    cache_dirpath = YLIR_CACHE_ROOT.joinpath(f'retrieve_hf')
    simple_model_infos_cache_dirpath = cache_dirpath.joinpath(f'simple_model_infos')

    logger.info(f' v Retrieving All Simple Model Infos ...')
    chunks_of_simple_model_infos = CachedChunks(
        simple_model_infos_cache_dirpath,
        get_all_data_from_huggingface_hub_api(f'{models_path}', params=dict(sort='lastModified', expand=['cardData', 'lastModified', 'likes', 'downloadsAllTime', 'siblings', 'tags']), token=token),
        number_per_file
    )
    logger.info(f' ^ Total = {len(chunks_of_simple_model_infos)}.')

    logger.info(f' v Retrieving All Model Infos ...')
    with tqdm.tqdm(initial=chunks_of_simple_model_infos.current_position, total=len(chunks_of_simple_model_infos), desc='Retrieve Model') as progress_bar:
        for chunk_of_simple_model_infos in chunks_of_simple_model_infos:
            model_infos_per_file = list()
            if worker_number is None:
                for simple_model_info in chunk_of_simple_model_infos:
                    model_id = simple_model_info['id']
                    model_storage = get_huggingface_hub_model_storage(model_id, simple=True, token=token)
                    progress_bar.set_description(f'Retrieve Model - {model_id}')
                    simple_model_info['usedStorage'] = model_storage
                    model_infos_per_file.append(simple_model_info)
                    progress_bar.update(1)
            else:
                with multiprocessing.Pool(worker_number) as pool:
                    logger.info(f' - Assign Tasks ... ')
                    model_storages = [pool.apply_async(get_huggingface_hub_model_storage, (simple_model_info['id'], True, token)) for simple_model_info in chunk_of_simple_model_infos]

                    logger.info(f' - Get Results ... ')
                    for model_storage, simple_model_info in zip(model_storages, chunk_of_simple_model_infos):
                        model_storage = model_storage.get()
                        model_id = simple_model_info['id']
                        progress_bar.set_description(f'Retrieve Model - {model_id}')
                        simple_model_info['usedStorage'] = model_storage
                        model_infos_per_file.append(simple_model_info)
                        progress_bar.update(1)

            save_filepath = save_dirpath.joinpath(f'huggingface_hub_model_infos_{chunks_of_simple_model_infos.current_chunk_id}.json')
            save_json(model_infos_per_file, save_filepath, indent=2)
            logger.info(f'Total {len(model_infos_per_file)} Model Info Items Saved In: \'{save_filepath}\'.')

    logger.info(f' ^ Retrieved.')


def get_huggingface_hub_metric_infos(save_dirpath: pathlib.Path, token: str | None = None):
    metrics_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/metrics'
    metrics = get_all_data_from_huggingface_hub_api(metrics_path, token=token)
    metric_infos = [metric for metric in metrics]
    save_filepath = save_dirpath.joinpath('huggingface_metric_infos.json')
    save_json(metric_infos, save_filepath, indent=2)
    logger.info(f'Total {len(metric_infos)} Metric Infos. Results Saved In: \'{save_filepath}\'.')


def get_huggingface_hub_task_infos(save_dirpath: pathlib.Path, token: str | None = None):
    tasks_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/tasks'
    tasks = get_all_data_from_huggingface_hub_api(tasks_path, token=token)
    task_infos = [task_info for task_id, task_info in tasks.items()]
    save_filepath = save_dirpath.joinpath('huggingface_task_infos.json')
    save_json(task_infos, save_filepath, indent=2)
    logger.info(f'Total {len(task_infos)} Tasks Infos. Results Saved In: \'{save_filepath}\'.')


####################################################################################################
# vvv Below is the functions to handle the files of models in local or remote Hugging Face Hub vvv #
####################################################################################################


def get_huggingface_hub_model_storage(model_id: str, simple: bool = True, token: str | None = None) -> int:
    if simple:
        model_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models/{model_id}'
        simple_model_info = get_one_data_from_huggingface_hub_api(model_path, params=dict(expand=['usedStorage']), token=token)
        if simple_model_info is None:
            model_storage = 0
        else:
            model_storage = simple_model_info['usedStorage']
    else:
        hf_file_system = HfFileSystem(token=token)
        filenames = list()
        filenames.extend(hf_file_system.glob(model_id+'/*.bin'))
        filenames.extend(hf_file_system.glob(model_id+'/*.pb'))
        filenames.extend(hf_file_system.glob(model_id+'/*.h5'))
        filenames.extend(hf_file_system.glob(model_id+'/*.hdf5'))
        filenames.extend(hf_file_system.glob(model_id+'/*.ckpt'))
        filenames.extend(hf_file_system.glob(model_id+'/*.keras'))
        filenames.extend(hf_file_system.glob(model_id+'/*.msgpack'))
        filenames.extend(hf_file_system.glob(model_id+'/*.safetensors'))
        model_storage = 0
        for filename in filenames:
            meta_data = get_hf_file_metadata(hf_hub_url(repo_id=model_id, filename=filename[len(model_id)+1:]))
            model_storage += meta_data.size
    return model_storage


def get_huggingface_hub_model_siblings(model_id: str, folder: str | None = None, suffixes: list[str] | None = None, simple: bool = True, token: str | None = None) -> list[tuple[str, str]]:
    if simple:
        model_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models/{model_id}'
        simple_model_info = get_one_data_from_huggingface_hub_api(model_path, params=dict(expand=['siblings']), token=token)
        model_siblings = list()
        if simple_model_info is not None:
            for sibling_detail in simple_model_info['siblings']:
                if suffixes is None:
                    model_siblings.append(sibling_detail['rfilename'])
                else:
                    if os.path.splitext(sibling_detail['rfilename'])[1] in suffixes:
                        model_siblings.append(sibling_detail['rfilename'])
    else:
        hf_file_system = HfFileSystem(token=token)
        model_siblings = list()
        for dirpath, dirnames, filenames in hf_file_system.walk(model_id):
            for filename in filenames:
                if suffixes is None:
                    model_siblings.append('/'.join([dirpath[len(model_id)+1:], filename]))
                else:
                    if os.path.splitext(filename)[1] in suffixes:
                        model_siblings.append('/'.join([dirpath[len(model_id)+1:], filename]))
    if folder is not None:
        model_siblings = [model_sibling for model_sibling in model_siblings if re.match(folder, model_sibling)]
    return model_siblings


def clean_huggingface_hub_root_cache(cache_dirpath: pathlib.Path):
    info = scan_cache_dir(cache_dirpath)
    commit_hashes = list()
    for repo in list(info.repos):
        for revision in list(repo.revisions):
            commit_hashes.append(revision.commit_hash)
    delete_strategy = info.delete_revisions(*commit_hashes)
    delete_strategy.execute()


def clean_huggingface_hub_model_cache(model_id: str, specify_cache_dirpath: pathlib.Path):
    default_cache_dirpath = pathlib.Path(HUGGINGFACE_HUB_CACHE)

    model_id = model_id.replace("/", "--")

    model_default_cache_dirpath = default_cache_dirpath.joinpath(f"models--{model_id}")
    if model_default_cache_dirpath.is_dir():
        delete_dir(model_default_cache_dirpath)

    model_specify_cache_dirpath = specify_cache_dirpath.joinpath(f"models--{model_id}")
    if model_specify_cache_dirpath.is_dir():
        delete_dir(model_specify_cache_dirpath)


##############################################################################################
# vvv Below is the functions to handle the README.md files of models in Hugging Face Hub vvv #
##############################################################################################


def filter_readme_filepaths(filepaths: list[str]) -> list[str]:
    readme_filepaths = list()
    pattern = re.compile(r'.*readme(?:\.[^/\\]*)?$', re.IGNORECASE)
    for filepath in filepaths:
        if re.match(pattern, filepath) is not None:
            readme_filepaths.append(filepath)

    return readme_filepaths


def get_huggingface_hub_model_readme(model_id: str, token: str | None = None) -> str:
    hf_file_system = HfFileSystem(token=token)
    if hf_file_system.exists(f'{model_id}/README.md'):
        try:
            readme_file_size = hf_file_system.size(f'{model_id}/README.md')
            if 1*1024*1024*1024 < readme_file_size:
                logger.error(f"REPO: {model_id}. Strange README File Error - File Size To Large: {get_human_readable_size_representation(readme_file_size)}")
                raise MemoryError
            with hf_file_system.open(f'{model_id}/README.md', mode='r', encoding='utf-8') as readme_file:
                readme = readme_file.read()
                readme = readme.replace('\t', ' ')
                return readme
        except UnicodeDecodeError as error:
            logger.error(f"REPO: {model_id}. Encoding Error - The Encoding [UTF-8] are Invalid. - Error: {error}")
            raise error
        except Exception as exception:
            logger.error(f"REPO: {model_id}. Encounter An Error {exception}.")
            raise exception
    else:
        logger.info(f"REPO: {model_id}. No README.md, skip.")
        raise FileNotFoundError


def extract_possible_metrics_from_readme(readme: str) -> dict[str, list[str] | list[dict[str, list[str]]]]:
    possible_metrics: dict[str, list[str] | list[dict[str, list[str]]]] = dict()

    yamlfm, readme = split_front_matter_from_readme_string(readme)

    possible_metrics['table_related'] = extract_possible_tables_from_readme_string(readme)

    readme = re.sub(README_TABLE_Pattern, '', readme)
    readme = re.sub(README_DATE_Pattern, '', readme)
    readme = re.sub(README_DATETIME_Pattern, '', readme)

    possible_metrics['digit_related'] = extract_possible_digits_from_readme_string(readme)
    
    return possible_metrics


##############################################################################################
# vvv Below is the functions to handle the README.md files of models in Hugging Face Hub vvv #
##############################################################################################


def infer_supported_frameworks(model_info: dict) -> list[Literal['optimum', 'onnx', 'keras', 'tflite']]:
    all_supported_frameworks = set(['optimum', 'onnx', 'keras', 'tflite'])
    tags = set(model_info['tags'])
    if len(tags & set(['transformers', 'diffusers', 'timm', 'sentence-transformers'])) == 0:
        for sibling in model_info['siblings']:
            filename = pathlib.Path(sibling['rfilename'])
            if filename.suffix in ['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']:
                tags.add('keras')

            if filename.suffix in ['.tflite']:
                tags.add('tflite')

            if filename.suffix in ['.onnx']:
                tags.add('onnx')

    else:
        tags.add('optimum')

    supported_frameworks = list(tags & all_supported_frameworks)
    return supported_frameworks
