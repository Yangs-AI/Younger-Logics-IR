#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 16:04:03
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-23 14:15:18
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commands import equip_logger


@click.group(name='create')
def create():
    pass


@create.group(name='onnx')
def create_onnx():
    pass


@create_onnx.group(name='retrieve')
def create_onnx_retrieve():
    pass


@create_onnx_retrieve.command(name='huggingface')
@click.option('--mode',             required=True,  type=click.Choice(['Model_Infos', 'Model_IDs', 'Metrics', 'Tasks'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from Huggingface.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--library',          required=False, type=str, default=None, help='The library type to which the model belongs, used to filter all model information containing this tag (library). If None, retrieve without any filter options.')
@click.option('--token',            required=False, type=str, default=None, help='The HuggingFace token, which requires registering an account on HuggingFace and manually setting the access token. If None, retrieve without HuggingFace access token.')
@click.option('--worker-number',    required=False, type=int, default=10, help='Increase this parameter when retrieving large-scale data to speed up progress. If not specified, 10 threads will be used.')
@click.option('--mirror-url',       required=False, type=str, default='', help='The URL of the HuggingFace mirror site, which may sometimes speed up your data retrieval process, but this tools cannot guarantee data integrity of the mirror site. If not specified, use HuggingFace official site.')
@click.option('--force-reload',     is_flag=True,   help='Use to ignore previous download records and redownload after use.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_huggingface(
    mode,
    save_dirpath,
    library, token,
    worker_number,
    mirror_url,
    force_reload,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.huggingface import retrieve

    kwargs = dict(
        library=library,
        token=token,
        worker_number=worker_number,
        force_reload=force_reload,
    )

    retrieve.main(mode, save_dirpath, mirror_url, **kwargs)


@create_onnx_retrieve.command(name='onnx')
@click.option('--mode',             required=True,  type=click.Choice(['Model_Infos', 'Model_IDs'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from Huggingface.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--force-reload',     is_flag=True,   help='Use to ignore previous download records and redownload after use.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_onnx(
    mode,
    save_dirpath,
    force_reload,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.onnx import retrieve

    kwargs = dict(
        force_reload=force_reload,
    )

    retrieve.main(mode, save_dirpath, **kwargs)


@create_onnx_retrieve.command(name='torchvision')
@click.option('--mode',             required=True,  type=click.Choice(['Model_Infos', 'Model_IDs'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from Huggingface.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--force-reload',     is_flag=True,   help='Use to ignore previous download records and redownload after use.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_torchvision(
    mode,
    save_dirpath,
    force_reload,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.torchvision import retrieve

    kwargs = dict(
        force_reload=force_reload,
    )

    retrieve.main(mode, save_dirpath, **kwargs)


@create_onnx.group(name='convert')
def create_onnx_convert():
    pass


@create_onnx_convert.group(name='huggingface')
@click.option('--model-ids-filepath',   required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model IDs file, which is obtained using the command: `younger logics ir create onnx retrieve [hub_type] --mode Model_IDs ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--status-filepath',      required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='The file records the conversion status of the models that have already been processed. If it is deleted, the processing progress will be lost.')
@click.option('--device',               required=False, type=click.Choice(['cpu', 'cuda'], case_sensitive=True), default='cpu', help='Used to indicate whether to use GPU or CPU when converting models.')
@click.option('--framework',            required=False, type=click.Choice(['optimum', 'onnx', 'keras', 'tflite'], case_sensitive=True), default='optimum', help='Indicates the framework to which the model belonged prior to conversion.')
@click.option('--model-size-threshold', required=False, type=int, default=3*1024*1024*1024, help='Used to filter out oversized models to prevent process interruptions due to excessive storage usage. (Note: The storage space occupied by models is a simple estimation and may have inaccuracies. Please use with caution.)')
@click.option('--token',                required=False, type=str, default=None, help='The HuggingFace token, which requires registering an account on HuggingFace and manually setting the access token. If None, retrieve without HuggingFace access token.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_huggingface(
    model_ids_filepath,
    save_dirpath, cache_dirpath,
    status_filepath,
    device, framework, model_size_threshold, token,
    logging_filepath
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.huggingface import convert

    convert.main(model_ids_filepath, save_dirpath, cache_dirpath, status_filepath, device=device, framework=framework, model_size_threshold=model_size_threshold, token=token)


@create_onnx_convert.group(name='onnx')
@click.option('--model-ids-filepath',   required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model IDs file, which is obtained using the command: `younger logics ir create onnx retrieve [hub_type] --mode Model_IDs ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--status-filepath',      required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='The file records the conversion status of the models that have already been processed. If it is deleted, the processing progress will be lost.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_onnx(
    model_ids_filepath,
    save_dirpath, cache_dirpath, status_filepath,
    logging_filepath
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.onnx import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, status_filepath)


@create_onnx_convert.group(name='torchvision')
@click.option('--model-ids-filepath',   required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model IDs file, which is obtained using the command: `younger logics ir create onnx retrieve [hub_type] --mode Model_IDs ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--status-filepath',      required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='The file records the conversion status of the models that have already been processed. If it is deleted, the processing progress will be lost.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_torchvision(
    model_ids_filepath,
    save_dirpath, cache_dirpath, status_filepath,
    logging_filepath
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.create.torchvision import convert

    convert.main(save_dirpath, cache_dirpath, model_ids_filepath, status_filepath)


@create.group(name='core')
def create_core():
    pass


@create_core.group(name='convert')
def create_core_convert():
    pass