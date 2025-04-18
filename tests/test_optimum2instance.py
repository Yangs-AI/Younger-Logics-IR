#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-10 21:08:41
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-24 09:44:54
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger.commons.io import create_dir

from younger_logics_ir.commons.constants import YLIROriginHub
from younger_logics_ir.modules import LogicX, Implementation, Origin
from younger_logics_ir.scripts.hubs.huggingface.convert import convert_onnx

#from optimum.exporters.onnx import main_export


@click.command()
@click.option('--cache-dirpath', required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--print-dirpath', required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Print directory, where data is saving.')
def main(cache_dirpath, print_dirpath):

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFOfc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFCvt')
    create_dir(cvt_cache_dirpath)

    #model_id = 'hks1444/bert-base-turkish-cased2_with_categories'
    model_id = 'csukuangfj/sherpa-onnx-lstm-zh-2023-02-20'
    # main_export(model_id, cvt_cache_dirpath, opset=14, device='cpu', cache_dir=ofc_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
    status, instances, filenames = convert_onnx(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device='cpu')
    #status, instances = convert_optimum(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device='cpu')
    print(status)
    print(instances)
    print(filenames)
    for instance, filename in zip(instances, filenames):
        instance.insert_label(
            Implementation(
                origin=Origin(YLIROriginHub.HUGGINGFACE, 'csukuangfj', 'sherpa-onnx-lstm-zh-2023-02-20' + '-F-' + filename),
                like=0,
                download=0,
            )
        )
        instance.save(print_dirpath.joinpath(f'{instance.unique}'))
        #print(LogicX.luid(instance.logicx))


if __name__ == '__main__':
    main()
