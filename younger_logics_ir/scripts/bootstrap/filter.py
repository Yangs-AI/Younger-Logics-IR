#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-16 08:58:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:55:09
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib
import networkx
import multiprocessing

from younger.commons.io import create_dir, save_json

from younger_logics_ir.modules import Instance, LogicX, Origin
from younger_logics_ir.commons.logging import logger


def get_opset_version(opset_import: dict[str, int]) -> int | None:
    opset_version = opset_import.get('', None)
    return opset_version


def check_instance(parameter: tuple[pathlib.Path, int]) -> pathlib.Path | None:
    path, opset_version = parameter
    instance = Instance()
    try:
        instance.load(path)

        if opset_version is not None and opset_version != get_opset_version(instance.logicx.dag.graph['opset_import']):
            return None
        else:
            return path
    except:
        return None


def filter_instance(instance_dirpath: pathlib.Path, standard_dirpath: pathlib.Path, skeleton_dirpath: pathlib.Path) -> tuple[Origin, int, bool, bool, str, networkx.DiGraph]:
    instance = Instance()
    instance.load(instance_dirpath)
    instance_unique = instance.unique

    logicx, logicx_sods = LogicX.simplify(instance.logicx)
    org_logicxs = [logicx] + logicx_sods
    family = networkx.DiGraph()

    for org_logicx in org_logicxs:
        std_logicx = LogicX.standardize(org_logicx)
        skt_logicx = LogicX.skeletonize(org_logicx)

        std_logicx_id = LogicX.hash(std_logicx)
        skt_logicx_id = LogicX.hash(skt_logicx)

        org_logicx_id = LogicX.luid(org_logicx)
        family.add_edge(org_logicx.relationship, org_logicx_id, standard=std_logicx_id, skeleton=skt_logicx_id)

        std_logicx_savepath = standard_dirpath.joinpath(std_logicx_id)
        if not std_logicx_savepath.is_file():
            std = True
            std_logicx.save(std_logicx_savepath)
        else:
            std = False

        skt_logicx_savepath = skeleton_dirpath.joinpath(skt_logicx_id)
        if not skt_logicx_savepath.is_file():
            skt = True
            skt_logicx.save(skt_logicx_savepath)
        else:
            skt = False

    return (instance.labels[0].origin, len(org_logicxs), std, skt, instance_unique, family)


def main(input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, opset_version: int | None = None, worker_number: int = 4):
    if opset_version:
        logger.info(f'Filter {opset_version} ONNX OPSET Version')
    else:
        logger.info(f'Filter All. ONNX OPSET Version Not Specified.')

    check_parameters = list()
    for input_dirpath in input_dirpaths:
        logger.info(f'Scanning Instances Directory Path: {input_dirpath}')
        for instance_dirpath in input_dirpath.iterdir():
            check_parameters.append((instance_dirpath, opset_version))

    standard_dirpath = output_dirpath.joinpath('standard')
    skeleton_dirpath = output_dirpath.joinpath('skeleton')
    create_dir(standard_dirpath)
    create_dir(skeleton_dirpath)

    logger.info(f'Total Instances To Be Filtered: {len(check_parameters)}')
    instance_dirpaths = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(check_parameters), desc=f'Initial Filter - For OPSET={"ALL" if opset_version is None else opset_version}') as progress_bar:
            for index, path in enumerate(pool.imap_unordered(check_instance, check_parameters), start=1):
                if path is not None:
                    instance_dirpaths.append(path)
                progress_bar.update(1)
    logger.info(f'Total Instances After Initial Opset Filter: {len(instance_dirpaths)}')

    logger.info(f'Total Instances To Be Simplified - Standardize & Skeletonize: {len(instance_dirpaths)}')
    instance_count = 0
    standard_count = 0
    skeleton_count = 0
    pedigree: dict[str, networkx.DiGraph] = dict()
    with tqdm.tqdm(total=len(instance_dirpaths), desc='Simplify - Standardize & Skeleonize') as progress_bar:
        for index, instance_dirpath in enumerate(instance_dirpaths, start=1):
            origin, lgx_count, std, skt, instance_unique, family = filter_instance(instance_dirpath, standard_dirpath, skeleton_dirpath)
            instance_count += lgx_count
            standard_count += std
            skeleton_count += skt
            pedigree[instance_unique] = networkx.readwrite.json_graph.adjacency_data(family)
            progress_bar.set_postfix({f'Current Model ID': f'{origin.hub}/{origin.owner}/{origin.name} - {lgx_count}'})
            progress_bar.update(1)
    logger.info(f'Total/Standard/Skeleton: {instance_count}/{standard_count}/{skeleton_count}')

    pdg_filepath = output_dirpath.joinpath('pedigree.json')
    save_json(pedigree, pdg_filepath, indent=2)
    logger.info(f'Finished')
