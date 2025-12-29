#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-02-11 11:20:00
# Author: Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-12 17:39:02
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


'''
This project is based on the repositorys "ppuda" (https://github.com/facebookresearch/ppuda), 
with modifications for converting DeepNets-1M models to Younger instances.
'''

import os
import onnx
import torch
import tqdm
import click
import pathlib
import multiprocessing

from typing import Any
from google.protobuf import text_format

from ppuda.deepnets1m.net import Network

from younger.commons.io import loads_json, saves_json, create_dir
from younger.commons.logging import set_logger, use_logger, logger

from younger_logics_ir.modules import Instance, Origin, Implementation
from younger_logics_ir.converters import convert
from younger_logics_ir.commons.constants import YLIROriginHub


def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


def convert_pipeline(params):
    model_id, net_args, cvt_cache_dirpath, instances_dirpath, opset = params
    onnx_model_filepath = cvt_cache_dirpath.joinpath(f'{model_id}.onnx')
    try:
        # Create network
        dummy_input = torch.randn(1, 3, 32, 32) 
        net = Network(num_classes=10,
                    is_imagenet_input=False,
                    **net_args).eval()

        # Convert the network to ONNX
        torch.onnx.export(net, dummy_input, onnx_model_filepath, verbose=False, opset_version=opset)

        # Convert the ONNX model to YLIR instance
        onnx_model = onnx.load(onnx_model_filepath)
        instance = Instance()
        instance.setup_logicx(convert(onnx_model))
        instance.insert_label(
            Implementation(
                origin = Origin(YLIROriginHub.NAS, f'DeepNets-1M', model_id)
            )
        )
        instance.save(instances_dirpath.joinpath(instance.unique))
        onnx_model_filepath.unlink(missing_ok=True)

        return True, model_id 
    except Exception as exception:
        print(f'Error: {model_id} - {exception}')
        return False, model_id


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, start_index: int, end_index: int) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    status_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}.sts')
    if status_filepath.is_file():
        with open(status_filepath, 'r') as status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, start_index: int, end_index: int, convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


@click.command()
@click.option('--model-infos-filepath', required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=str), help='The Dir specifies the address of the Model Infos file, which is obtained using the command: `younger logics ir create onnx retrieve huggingface --mode Model_Infos ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--opset',                required=True,  type=int, help='Used to indicate which opset version the model needs to be converted to in the ONNX format.')
@click.option('--start-index',          required=False, type=int, default=None, help='Used to indicate the position of the model in the model_infos list where the conversion starts.')
@click.option('--end-index',            required=False, type=int, default=None, help='Used to indicate the position of the model in the model_infos list where the conversion ends (the index are excluded).')
@click.option('--worker-number',        required=False, type=int, default=1, help='Used to indicate how many processes are concurrently performing the model conversion tasks.')
def main(
    model_infos_filepath,
    save_dirpath, cache_dirpath,
    opset,
    start_index, end_index,
    worker_number,
):
    set_logger(f'DeepNets-1M_Convert', mode='console', level='INFO')
    use_logger(f'DeepNets-1M_Convert')

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-DN1MCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-DN1MSts')
    create_dir(sts_cache_dirpath)

    model_infos = torch.load(model_infos_filepath, weights_only=False)
    assert isinstance(model_infos, list)

    start_index = start_index or 0
    end_index = end_index or len(model_infos)
    print('length of model_infos: ', len(model_infos))
    model_infos = model_infos[start_index:end_index]

    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, start_index, end_index)

    logger.info(f'Packaging Parameters For Conversion ...')
    params = list()
    with tqdm.tqdm(total=len(model_infos), desc='Packaging Params') as progress_bar:
        for (idx, net_args) in model_infos:
            model_id = str(idx)
            if last_handled_model_id is not None:
                if model_id == last_handled_model_id:
                    last_handled_model_id = None
                progress_bar.set_description(f'Converted, Skip - {model_id}')
                progress_bar.update(1)
                continue
            params.append((model_id, net_args, cvt_cache_dirpath, instances_dirpath, opset))
            progress_bar.update(1)

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(params), desc='Converting Models') as progress_bar:
            for index, (status, model_id) in enumerate(pool.imap(convert_pipeline, params), start=1):
                set_convert_status_last_handled_model_id(sts_cache_dirpath, start_index, end_index, status, model_id)
                progress_bar.set_description(f'Converting - {model_id}')
                progress_bar.update(1)


if __name__ == '__main__':
    main()
