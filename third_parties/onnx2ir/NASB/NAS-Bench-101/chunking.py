#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-06 21:31:22
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-06 21:51:59
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger


@click.command()
@click.option('--load-filepath', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model Infos file.')
@click.option('--save-filepath', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--size-of-chunk', required=True, type=int, help='Chunking Size')
def main(load_filepath: pathlib.Path, save_filepath: pathlib.Path, size_of_chunk: int):
    model_infos = load_json(load_filepath)
    for index, start_index in enumerate(range(0, len(model_infos), size_of_chunk)):
        chunk = model_infos[start_index:start_index+size_of_chunk]
        save_json(chunk, save_filepath.with_suffix(f'.{index}'))
        logger.info(f'Chunk - {index} - Saved.')
    logger.info(f'Done.')


if __name__ == '__main__':
    main()
