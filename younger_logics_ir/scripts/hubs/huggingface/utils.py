#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-09-09 11:37:47
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import time
import tqdm
import pathlib
import requests

from typing import Any, Literal, Generator
from huggingface_hub import utils, HfFileSystem, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from younger.commons.io import load_json, save_json, save_pickle, load_pickle, delete_dir, get_human_readable_size_representation
from younger.commons.cache import CachedChunks
from younger.commons.string import extract_possible_digits_from_readme_string, extract_possible_tables_from_readme_string, split_front_matter_from_readme_string, README_DATE_Pattern, README_DATETIME_Pattern, README_TABLE_Pattern

from younger_logics_ir.commons.cache import YLIR_CACHE_ROOT
from younger_logics_ir.commons.logging import logger


HUGGINGFACE_HUB_API_ENDPOINT = 'https://huggingface.co/api'

# Rate limit configuration
# Default: 1000 requests per 5 minutes (300s) => ~3.33 req/s
# Maintain ~90% of limit to stay safe: 300s / (limit * 0.9)
_rate_limit = 1000
_last_api_request_time = 0.0
_api_request_interval_sec = 300.0 / (_rate_limit * 0.9)

def set_rate_limit(rate_limit: int):
    """Configure the rate limit and recalculate request interval.

    :param rate_limit: Maximum number of requests allowed per 5 minutes
    """
    global _rate_limit, _api_request_interval_sec
    _rate_limit = rate_limit
    # Use 90% of the limit to maintain safety margin
    _api_request_interval_sec = 300.0 / (_rate_limit * 0.9)
    logger.info(f"Rate limit set to {_rate_limit} requests/5min. Request interval: {_api_request_interval_sec:.3f}s (~{1/_api_request_interval_sec:.2f} req/s)")


def _apply_rate_limit():
    """Apply request rate limiting to stay under configured API limits."""
    global _last_api_request_time
    elapsed = time.time() - _last_api_request_time
    if elapsed < _api_request_interval_sec:
        time.sleep(_api_request_interval_sec - elapsed)
    _last_api_request_time = time.time()


def _extract_rate_limit_reset_time(headers: dict) -> float | None:
    """Extract the rate limit reset time in seconds from response headers.

    Tries RateLimit-Reset-After (in seconds) first, then falls back to Retry-After.

    :param headers: Response headers dictionary
    :return: Seconds to wait, or None if not found
    """
    # Try HuggingFace's RateLimit-Reset-After header (seconds)
    reset_after = headers.get('RateLimit-Reset-After')
    if reset_after is not None:
        try:
            return float(reset_after)
        except (ValueError, TypeError):
            pass

    # Fall back to standard Retry-After header
    retry_after = headers.get('Retry-After')
    if retry_after is not None:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass

    return None


def is_permanent_error(error_message: str) -> bool:
    """Return True if the error is permanent -- no opset version change can fix it.

    Matches by exception type name (everything before ': '). Exception types that
    indicate model-loading / config / file-structure issues are permanent because
    they happen before ONNX export uses the opset version.
    """
    _PERMANENT_TYPES = frozenset({
        'FileNotFoundError',
        'OSError',
        'ValueError',
        'ImportError',
        'TypeError',
    })
    colon_pos = error_message.find(': ')
    if colon_pos == -1:
        return False
    return error_message[:colon_pos] in _PERMANENT_TYPES


def get_minimum_opset_from_error(error_message: str) -> int | None:
    """Extract the minimum opset version required by an UnsupportedOperatorError.

    Returns None if the error message does not specify a target version.
    """
    # e.g. "Support for this operator was added in version 14"
    match = re.search(
        r'Support for this operator was added in version (\d+)',
        error_message,
    )
    if match:
        return int(match.group(1))
    # e.g. "Please try opset version 11"
    match = re.search(
        r'please try opset version (\d+)',
        error_message,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def _http_request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> requests.Response | None:
    """
    HTTP request with timeout and retry on transient errors.

    Retries on: Timeout, ConnectionError (up to max_retries times).
    On HTTP 429: parses RateLimit header, waits, then retries once.
    All other errors return None immediately.

    :param session: requests Session to use
    :param method: HTTP method (GET, POST, etc.)
    :param url: Request URL
    :param max_retries: Maximum number of retries for Timeout/ConnectionError (default 3)
    :param kwargs: Passed through to session.request()
    :return: Response object or None on failure
    """
    kwargs.setdefault('timeout', 30)
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            _apply_rate_limit()
            response = session.request(method, url, **kwargs)
            utils.hf_raise_for_status(response)
            return response
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"HTTP timeout for '{url}' (attempt {attempt+1}/{max_retries+1}): {e}")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"HTTP connection error for '{url}' (attempt {attempt+1}/{max_retries+1}): {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 429:
                reset_time = _extract_rate_limit_reset_time(e.response.headers)
                if reset_time is not None and reset_time > 0:
                    logger.warning(f"HTTP 429 for '{url}', waiting {reset_time:.1f}s as per RateLimit header...")
                    time.sleep(reset_time)
                    # Single retry after waiting
                    try:
                        response = session.request(method, url, **kwargs)
                        utils.hf_raise_for_status(response)
                        return response
                    except Exception as retry_e:
                        logger.error(f"HTTP 429 retry failed for '{url}': {retry_e}")
                        return None
                else:
                    logger.warning(f"HTTP 429 for '{url}' but no valid RateLimit header. Aborting.")
                    time.sleep(300) # usually a time reset takes 5 min. just in case we hit the worst situation.
                    return None
            else:
                logger.error(f"HTTP {status} for '{url}': {e}")
                return None
        except utils.HfHubHTTPError as e:
            logger.error(f"HF Hub error for '{url}': {e.request_id} - {e.server_message}")
            return None
        if attempt < max_retries:
            wait = 2 ** attempt
            logger.info(f"Retrying '{url}' in {wait:.1f}s...")
            time.sleep(wait)
    logger.error(f"All {max_retries+1} attempts failed for '{url}': {last_exception}")
    return None


def get_one_data_from_huggingface_hub_api(path: str, params: dict | None = None, token: str | None = None) -> Any:
    """
    Fetch single data from Hugging Face API with rate limiting and timeout.

    On transient errors (timeout, connection error), retries up to 3 times.
    On HTTP 429, parses RateLimit header, waits, then retries once.
    On permanent errors (404, 401, etc.), returns None immediately.

    :param path: API endpoint
    :param params: Query parameters
    :param token: Hugging Face token
    :return: Response JSON or None on error
    """
    params = params or dict()
    # Enforce limit=1000 to maximize items per request
    params['limit'] = 1000

    session = requests.Session()
    headers = utils.build_hf_headers(token=token)

    response = _http_request_with_retry(session, 'GET', path, params=params, headers=headers, timeout=30)
    if response is None:
        return None
    return response.json()


def get_all_data_from_huggingface_hub_api(path: str, params: dict | None = None, token: str | None = None) -> Generator[Any, None, None]:
    """
    Paginate through Hugging Face API with rate limiting and timeout.

    On transient errors (timeout, connection error), retries up to 3 times
    per page request. On HTTP 429, parses RateLimit header, waits, then retries once.
    On permanent errors, stops pagination.

    :param path: API endpoint path
    :param params: Query parameters
    :param token: Hugging Face token
    :yield: Items from all pages
    """
    params = params or dict()
    # Enforce limit=1000 to maximize items per request
    params['limit'] = 1000

    session = requests.Session()
    headers = utils.build_hf_headers(token=token)

    # First page
    response = _http_request_with_retry(session, 'GET', path, params=params, headers=headers, timeout=30)
    if response is None:
        return
    yield from response.json()
    next_page_path = response.links.get("next", {}).get("url")

    # Follow pages
    while next_page_path is not None:
        logger.debug(f"Pagination: fetching next page")
        response = _http_request_with_retry(session, 'GET', next_page_path, headers=headers, timeout=30)
        if response is None:
            return
        yield from response.json()
        next_page_path = response.links.get("next", {}).get("url")


#############################################################################################################
# vvv Below is the functions to get the information of models, metrics, and tasks from Hugging Face Hub vvv #
#############################################################################################################


def get_huggingface_hub_model_ids(token: str | None = None) -> Generator[dict, None, None]:
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'
    # Fetch models with minimal payload: no expand, just sort/direction (limit set by pagination)
    # Returns full model objects (id, lastModified, etc.) to preserve metadata without extra API calls
    models = get_all_data_from_huggingface_hub_api(models_path, params=dict(sort='lastModified', direction=-1), token=token)
    return models


def _is_chunk_completed(filepath: pathlib.Path) -> bool:
    """Check if a saved JSON file already contains usedStorage data by peeking at the header."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read first 2KB — the first model's 'usedStorage' key should be within this range
            head = f.read(2048)
            return '"usedStorage"' in head
    except Exception:
        return False


def _find_resume_chunk_id(save_dirpath: pathlib.Path, num_of_chunks: int) -> int | None:
    """
    Find the first chunk whose output file is missing or incomplete.

    Scans output files sequentially from file number 1 upward.
    Output files follow 1-based naming: file _{N}.json holds data
    from CachedChunks chunk N-1 (0-based).

    Returns the 0-based CachedChunks index of the first incomplete chunk,
    or None if all chunks have complete output files.
    """
    for chunk_index in range(num_of_chunks):
        file_number = chunk_index + 1
        filepath = save_dirpath.joinpath(
            f'huggingface_hub_model_infos_{file_number}.json'
        )
        if not (filepath.is_file() and _is_chunk_completed(filepath)):
            return chunk_index
    return None


def get_huggingface_hub_model_infos(save_dirpath: pathlib.Path, token: str | None = None, number_per_file: int | None = None, include_storage: bool = False):
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'

    cache_dirpath = YLIR_CACHE_ROOT.joinpath(f'retrieve_hf')
    simple_model_infos_cache_dirpath = cache_dirpath.joinpath(f'simple_model_infos')

    logger.info(f' v Retrieving All Simple Model Infos ...')
    chunks_of_simple_model_infos = CachedChunks(
        simple_model_infos_cache_dirpath,
        get_all_data_from_huggingface_hub_api(
            f'{models_path}',
            params=dict(sort='lastModified', expand=['cardData', 'createdAt', 'lastModified', 'likes', 'downloads', 'downloadsAllTime', 'siblings', 'evalResults', 'pipeline_tag', 'tags']),
            token=token,
        ),
        number_per_file
    )
    logger.info(f' ^ Total = {len(chunks_of_simple_model_infos)}.')

    if include_storage:
        logger.info(f' v Retrieving All Model Infos (Storage included) ...')
    else:
        logger.info(f' v Retrieving All Model Infos (Storage excluded) ...')

    # Resume support for storage retrieval: rewind to first incomplete chunk
    if include_storage:
        resume_index = _find_resume_chunk_id(save_dirpath, chunks_of_simple_model_infos._num_of_chunks)
        if resume_index is None:
            logger.info(f' All chunks already completed with storage data. Nothing to do.')
            logger.info(f' ^ Retrieved.')
            return
        if resume_index < chunks_of_simple_model_infos._current_index:
            logger.info(f' Resuming from chunk {resume_index} (files _1~_{resume_index} already have storage data)')
            # Update both the status file (for future restarts) and the in-memory state
            save_pickle(resume_index, simple_model_infos_cache_dirpath.joinpath(CachedChunks._status_cache_filename_))
            chunks_of_simple_model_infos._current_index = resume_index

    with tqdm.tqdm(initial=chunks_of_simple_model_infos.current_position, total=len(chunks_of_simple_model_infos), desc='Retrieve Model') as progress_bar:
        for chunk_of_simple_model_infos in chunks_of_simple_model_infos:
            save_filepath = save_dirpath.joinpath(f'huggingface_hub_model_infos_{chunks_of_simple_model_infos.current_chunk_id}.json')

            # Resume: skip chunks whose output files already contain storage data
            if include_storage and save_filepath.is_file() and _is_chunk_completed(save_filepath):
                logger.info(f'Skipping already completed chunk → {save_filepath.name}')
                progress_bar.update(len(chunk_of_simple_model_infos))
                continue

            model_infos_per_file = list()

            # NOTE:
            # We intentionally do NOT request usedStorage from the list endpoint.
            # In practice this field is not reliably available in list responses,
            # so include_storage must fetch usedStorage model-by-model for accuracy.
            for index, simple_model_info in enumerate(chunk_of_simple_model_infos):
                model_id = simple_model_info['id']

                if include_storage:
                    simple_model_info['usedStorage'] = get_huggingface_hub_model_storage(model_id, simple=True, token=token)

                # set_description is relatively expensive; update periodically.
                if index % 100 == 0:
                    progress_bar.set_description(f'Retrieve Model - {model_id}')

                model_infos_per_file.append(simple_model_info)
                progress_bar.update(1)

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
    task_infos = [task for task in tasks]
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
# ^^^ Above is the functions to handle the README.md files of models in Hugging Face Hub ^^^ #
##############################################################################################


def infer_supported_frameworks(model_info: dict) -> list[Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']]:
    all_supported_frameworks = set(['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'])
    all_supported_tags = set(['transformers', 'diffusers', 'timm', 'sentence-transformers', 'stable-baselines3'])
    frameworks = set([])
    tags = set(model_info['tags'])
    if len(tags & all_supported_tags) == 0:
        for sibling in model_info['siblings']:
            filename = pathlib.Path(sibling['rfilename'])
            if filename.suffix in ['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']:
                frameworks.add('keras')

            if filename.suffix in ['.tflite']:
                frameworks.add('tflite')

            if filename.suffix in ['.onnx']:
                frameworks.add('onnx')

    elif 'stable-baselines3' in tags:
        frameworks.add('stable_baselines3')

    else:
        frameworks.add('optimum')

    supported_frameworks = list(frameworks & all_supported_frameworks)
    return supported_frameworks
