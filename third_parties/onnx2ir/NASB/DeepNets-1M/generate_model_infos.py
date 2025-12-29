#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-12-24 12:34:52
# Author: Luzhou Peng(彭路洲) & Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-24 16:02:54
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


"""
To generate the model_infos for the DeepNets-1M dataset.
"""

import gc
import pathlib
import click
import torch
import tqdm
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.utils import set_seed

set_seed(1111) # To be consistent with config.py of ppuda


@click.command()
@click.option('--save-dirpath', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the model_infos will be saved.')
@click.option('--data-dirpath', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data is stored.')
@click.option('--split-number', required=False, type=int, default=200000, help='The interval at which the model_infos will be saved.')
def main(
    save_dirpath,
    data_dirpath,
    split_number,
):
    added_graphs = set()
    graphs_queue = DeepNets1M.loader(split='train',
                                    nets_dir=data_dirpath,
                                    large_images=False,
                                    virtual_edges=50,
                                    arch=None)

    quotient, remainder = divmod(len(graphs_queue), split_number)

    model_info_list = []
    print('len(graphs_queue): ', len(graphs_queue))

    point = 0
    points = list()
    for i in range(split_number):
        point = point + quotient + (1 if i < remainder else 0)
        points.append(point)
    print(points)

    split_id = 0
    for index, graphs in enumerate(tqdm.tqdm(graphs_queue)):
        if index >= len(graphs_queue):
            break
        net_args, net_idx = graphs.net_args[0], graphs.net_inds[0]

        if index >= points[split_id]:
            model_info_list.sort(key=lambda x: x[0])
            torch.save(model_info_list, save_dirpath.joinpath(f'DeepNets-1M_Model_Infos_{split_id}.pth'))
            print(f'Split No.{split_id} Saved')
            model_info_list.clear()
            split_id += 1
            gc.collect()

        model_info_list.append((net_idx, net_args))
        assert not net_idx in added_graphs
        added_graphs.add(net_idx)

    if len(model_info_list) != 0:
        model_info_list.sort(key=lambda x: x[0])
        torch.save(model_info_list, save_dirpath.joinpath(f'DeepNets-1M_Model_Infos_{split_id}.pth'))
        print(f'Split No.{split_id} Saved')
        model_info_list.clear()
        split_id += 1
        gc.collect()

    print('len(added_graps) and len(graphs_queue): ', len(added_graphs), ' and ', len(graphs_queue))
    assert len(added_graphs) == len(graphs_queue)

if __name__ == '__main__':
    main()
