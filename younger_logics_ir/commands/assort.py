#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 16:04:03
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-19 12:51:43
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commons.logging import equip_logger


@click.group(name='assort')
def assort():
    pass


@assort.command(name='visualize')
@click.option('--mode',             required=True,  type=click.Choice(['LogicX', 'DAG'], case_sensitive=True), help='Indicates the type of data that needs to be visualized.')
@click.option('--load-filepath',    required=True,  type=click.Path(exists=True,  file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--save-filepath',    required=True,  type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--simplify',         is_flag=True,   help='Use to simplify graph.')
@click.option('--skeleton',         is_flag=True,   help='Use to skeleton graph.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def assort_visualize(
    mode,
    load_filepath,
    save_filepath,
    simplify,
    skeleton,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.tools import visualize

    kwargs = dict(
        simplify=simplify,
        skeleton=skeleton
    )

    visualize.main(mode, load_filepath, save_filepath, **kwargs)

@assort.command(name='statistics')
@click.option('--input-names',      required=True,  type=str, multiple=True, help='The name of each dataset. Multiple names should correspond to multiple input directories. The order of names and directories should match. You can use this like: --input-names dataset1 --input-names dataset2')
@click.option('--input-dirpaths',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), multiple=True, help='The directory where the dataset will be loaded. Multiple directories should correspond to multiple input names. The order of names and directories should match. You can use this like: --input-dirpaths /path/to/dataset1 --input-dirpaths /path/to/dataset2')
@click.option('--output-dirpath',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved. All the output files corresponding to the input datasets will be saved in this directory.')
@click.option('--mode',             required=True,  type=click.Choice(['junior', 'senior', 'motif', 'edit'], case_sensitive=True), help='Indicates the type of data that needs to be statistically analyzed.')
@click.option('--sample-number',    required=False, type=int, default=None, help='Indicates the type of data that needs to be statistically analyzed.')
@click.option('--worker-number',    required=False, type=int, default=None, help='Number of workers for speeding up (only support modes: `motif`).')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_statistics(
    input_names,
    input_dirpaths,
    output_dirpath,
    mode,
    sample_number,
    worker_number,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.tools import statistics

    statistics.main(input_names, input_dirpaths, output_dirpath, mode, sample_number, worker_number)


@assort.command(name='semantics')
@click.option('--input-names',      required=True,  type=str, multiple=True, help='The name of each dataset. Multiple names should correspond to multiple input filepaths. The order of names and filepaths should match. You can use this like: --input-names dataset1 --input-names dataset2')
@click.option('--input-filepaths',  required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), multiple=True, help='The filepath where the embeddings of each dataset will be loaded. Multiple filepaths should correspond to multiple input names. The order of names and filepaths should match. You can use this like: --input-filepaths /path/to/dataset1.csv --input-filepaths /path/to/dataset2.csv')
@click.option('--output-dirpath',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved. All the output files corresponding to the input datasets will be saved in this directory.')
@click.option('--mode',             required=True,  type=click.Choice(['cos', 'mmd', 'cls'], case_sensitive=True), help='Indicates the type of data that needs to be semantically analyzed.')
@click.option('--standardize',      is_flag=True,   help='Indicates the type of data that needs to be semantically analyzed.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_statistics(
    input_names,
    input_filepaths,
    output_dirpath,
    mode,
    standardize,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.tools import semantics

    semantics.main(input_names, input_filepaths, output_dirpath, mode, standardize)
