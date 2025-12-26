#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-04-17 17:20:31
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:58:40
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################

# Luzhou Peng Added Multiprocessing Feature.


import tqdm
import numpy
import pandas
import random
import pathlib
import networkx
import multiprocessing

from typing import Any, Literal
from collections import Counter
from scipy.stats import entropy

from younger.commons.io import save_json, create_dir, get_object_with_sorted_dict

from younger_logics_ir.modules import LogicX
from younger_logics_ir.commons.json import YLIRJSONEncoder
from younger_logics_ir.commons.logging import logger


def extract_junior_statistics(datasets: dict[str, list[pathlib.Path]], output_dirpath: pathlib.Path):
    for dataset_name, logicx_filepaths in datasets.items():
        logger.info(f'Now - {dataset_name}')
        ne_with_max_non = dict(
            number_of_nodes=0,
            number_of_edges=0
        ) # The (number of nodes, number of edge) with max number of nodes.
        ne_with_max_noe = dict(
            number_of_nodes=0,
            number_of_edges=0
        ) # The (number of nodes, number of edge) with max number of edges.

        operator_details: dict[str, tuple[str, str, dict]] = dict()
        operator_occurence: dict[str, int] = dict()
        detailed_statistics: dict[str, Any] = dict()
        for logicx_filepath in tqdm.tqdm(logicx_filepaths):
            logicx = LogicX()
            logicx.load(logicx_filepath)
            logicx_hash = LogicX.hash(logicx)
            non = logicx.dag.number_of_nodes()
            noe = logicx.dag.number_of_edges()
            if (ne_with_max_non['number_of_nodes'] < non) or (ne_with_max_non['number_of_nodes'] == non and ne_with_max_non['number_of_edges'] < noe):
                ne_with_max_non = dict(
                    number_of_nodes = non,
                    number_of_edges = noe
                )
            if (ne_with_max_noe['number_of_edges'] < noe) or (ne_with_max_noe['number_of_edges'] == noe and ne_with_max_noe['number_of_nodes'] < non):
                ne_with_max_noe = dict(
                    number_of_nodes = non,
                    number_of_edges = noe
                )

            this_operator_details: dict[str, tuple[str, str, dict]] = dict()
            this_operator_occurence: dict[str, int] = dict()
            for operator_node_index in logicx.node_indices('operator'):
                node_features = logicx.node_features(operator_node_index)
                uuid = node_features['node_uuid']
                if uuid not in operator_details:
                    operator_details[uuid] = (node_features['node_tuid'], node_features['node_type'], node_features.get('node_attr', {}))
                operator_occurence[uuid] = operator_occurence.get(uuid, 0) + 1
                if uuid not in this_operator_details:
                    this_operator_details[uuid] = (node_features['node_tuid'], node_features['node_type'], node_features.get('node_attr', {}))
                this_operator_occurence[uuid] = this_operator_occurence.get(uuid, 0) + 1

            this_statistics = dict(
                number_of_nodes = non,
                number_of_edges = noe,
                operator_details = get_object_with_sorted_dict(this_operator_details),
                operator_occurence = get_object_with_sorted_dict(this_operator_occurence),
            )
            detailed_statistics[logicx_hash] = this_statistics

        
        logicx_junior_statistics = dict(
            detail = detailed_statistics,
            overall = dict(
                ne_with_max_non = ne_with_max_non,
                ne_with_max_noe = ne_with_max_noe,
                operator_details = get_object_with_sorted_dict(operator_details),
                operator_occurence = get_object_with_sorted_dict(operator_occurence),
            )
        )

        logicx_junior_statistics_filepath = output_dirpath.joinpath(f'{dataset_name}_logicx_junior_statistics.json')
        save_json(logicx_junior_statistics, logicx_junior_statistics_filepath, cls=YLIRJSONEncoder)
        logger.info(f'LogicX Junior Statistics ({dataset_name}) Saved into: {logicx_junior_statistics_filepath}')


def dag_widest_layer_width(dag: networkx.DiGraph):
    topological_order = list(networkx.topological_sort(dag))
    node2layer: dict[str, int] = dict()  # node -> layer
    for node_index in topological_order:
        predecessors = list(dag.predecessors(node_index))
        node2layer[node_index] = max([node2layer[predecessor] + 1 for predecessor in predecessors] + [0])

    layer_widths: dict[int, int] = dict()
    for layer in node2layer.values():
        layer_widths[layer] = layer_widths.get(layer, 0) + 1

    wl_width = max(layer_widths.values())
    return wl_width


def compute_statistics_embedding(dag: networkx.DiGraph):
    num_nodes = dag.number_of_nodes()
    num_edges = dag.number_of_edges()
    avg_idgrs = numpy.mean([node_idgr for node_index, node_idgr in dag.in_degree])
    avg_odgrs = numpy.mean([node_odgr for node_index, node_odgr in dag.out_degree])
    lp_length = networkx.dag_longest_path_length(dag)

    occurance = Counter([dag.nodes[node_index]['node_uuid'] for node_index in dag.nodes])
    frequency = numpy.array(list(occurance.values())) / len(occurance)
    o_entropy = entropy(frequency)

    op_count = len(occurance)
    wl_width = dag_widest_layer_width(dag)

    statistics_embedding = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_idgrs': avg_idgrs,
        'avg_odgrs': avg_odgrs,
        'lp_length': lp_length,
        'o_entropy': o_entropy,
        'wl_width': wl_width,
        'op_count': op_count,
    }
    return statistics_embedding


def extract_senior_statistics(datasets: dict[str, list[pathlib.Path]], output_dirpath: pathlib.Path):
    for dataset_name, logicx_filepaths in datasets.items():
        logger.info(f'Now - {dataset_name}')
        statistics_embeddings = list()
        with tqdm.tqdm(total=len(logicx_filepaths), desc=f"Extracting stats: {dataset_name}") as progress_bar:
            for logicx_filepath in logicx_filepaths:
                logicx = LogicX()
                logicx.load(logicx_filepath)
                statistics_embeddings.append(compute_statistics_embedding(logicx.dag))
                progress_bar.update(1)
        data_frame = pandas.DataFrame(statistics_embeddings)
        data_frame.to_csv(output_dirpath.joinpath(f'{dataset_name}_logicx_senior_statistics.csv'), index=False)


def extract_motif_statistics(datasets: dict[str, list[pathlib.Path]], output_dirpath: pathlib.Path, sample_number: int | None, worker_number: int | None):
    if sample_number is not None:
        datasets = {
            dataset_name: random.sample(logicx_filepaths, min(sample_number, len(logicx_filepaths)))
            for dataset_name, logicx_filepaths in datasets.items()
        }

    # For Space Efficiency: Scan 2 Rounds.
    candidate_motif_hashes = set()
    top_k = 200
    radii = [1,2]

    motif_hashes: set[str] = set()
    motif_lookup: dict[str, networkx.DiGraph] = dict()
    motif_number: dict[str, int] = dict()
    logger.info(f' - First Scanning ...')
    for dataset_name, logicx_filepaths in datasets.items():
        logger.info(f'Now - {dataset_name}')
        motif_count: dict[str, int] = dict()
        with tqdm.tqdm(total=len(logicx_filepaths), desc=f"Extracting stats: {dataset_name}") as progress_bar:
            for logicx_filepath in logicx_filepaths:
                logicx = LogicX()
                logicx.load(logicx_filepath)
                progress_bar.set_postfix({f'Current Hash | # Nodes': f'{logicx_filepath.name}/{len(logicx.dag.nodes)}'})
                nodes = random.sample(list(logicx.dag.nodes), min(1000, len(logicx.dag.nodes)))
                for node_index in nodes:
                    for radius in radii:
                        motif = networkx.ego_graph(logicx.dag, node_index, radius=radius, center=True, undirected=True)
                        motif_hash = networkx.weisfeiler_lehman_graph_hash(motif, edge_attr=None, node_attr='node_uuid', iterations=3, digest_size=16)
                        motif_count[motif_hash] = motif_count.get(motif_hash, 0) + 1
                progress_bar.update(1)

        top_k_hashes = sorted(motif_count.keys(), key=lambda x: motif_count[x], reverse=True)[:top_k]
        for candidate_top_k in candidate_top_ks:
            candidate_motif_hashes[candidate_top_k].update(top_k_hashes[:candidate_top_k])
        motif_hashes.update(top_k_hashes)
        motif_number[dataset_name] = sum(motif_count.values())

    for dataset_name, logicx_filepaths in datasets.items():
        logger.info(f'Now - {dataset_name}')
        motif_count: dict[str, int] = dict()
        with tqdm.tqdm(total=len(logicx_filepaths), desc=f"Extracting stats: {dataset_name}") as progress_bar:
            for logicx_filepath in logicx_filepaths:
                logicx = LogicX()
                logicx.load(logicx_filepath)
                progress_bar.set_postfix({f'Current Hash | # Nodes': f'{logicx_filepath.name}/{len(logicx.dag.nodes)}'})
                nodes = random.sample(list(logicx.dag.nodes), min(10000, len(logicx.dag.nodes)))
                for node_index in logicx.dag.nodes:
                    for radius in radii:
                        motif = networkx.ego_graph(logicx.dag, node_index, radius=radius, center=True, undirected=True)
                        motif_hash = networkx.weisfeiler_lehman_graph_hash(motif, edge_attr=None, node_attr='node_uuid', iterations=3, digest_size=16)
                        if motif_hash in motif_hashes:
                            motif_count[motif_hash] = motif_count.get(motif_hash, 0) + 1
                            motif_lookup[motif_hash] = motif_lookup.get(motif_hash, LogicX.saves_dag(motif))
                progress_bar.update(1)

        motif_statistics = list()
        for motif_hash in motif_hashes:
            occurence = motif_count.get(motif_hash, 0)
            frequency = occurence / motif_number[dataset_name]
            motif_statistics.append({
                "motif_hash": motif_hash,
                "occurence": occurence,
                "frequency": frequency,
            })
        data_frame = pandas.DataFrame(motif_statistics)
        data_frame.to_csv(output_dirpath.joinpath(f'{dataset_name}_logicx_motif_statistics.csv'), index=False)

    for candidate_top_k in candidate_top_ks:
        candidate_motif_hashes[candidate_top_k] = list(candidate_motif_hashes[candidate_top_k])

    save_json(motif_lookup, output_dirpath.joinpath(f'logicx_motif_lookup.json'), indent=2)
    save_json(motif_number, output_dirpath.joinpath(f'logicx_motif_number.json'), indent=2)
    save_json(candidate_motif_hashes, output_dirpath.joinpath(f'logicx_candidate_motif_hashes.json'), indent=2)


def extract_edit_distances(datasets: dict[str, list[pathlib.Path]], output_dirpath: pathlib.Path):
    sample_number = 100
    datasets = {
        dataset_name: random.sample(logicx_filepaths, min(sample_number, len(logicx_filepaths)))
        for dataset_name, logicx_filepaths in datasets.items()
    }

    for dataset_name, logicx_filepaths in datasets.items():
        logger.info(f'Now - {dataset_name}')
        logicxs: list[LogicX] = list()
        for logicx_filepath in logicx_filepaths:
            logicx = LogicX()
            logicx.load(logicx_filepath)
            logicxs.append(logicx)

        n = len(logicxs)
        edit_distances = numpy.zeros((n, n))
        with tqdm.tqdm(total=n*(n-1)/2, desc=f"Extracting stats: {dataset_name}") as progress_bar:
            for i in range(n):
                for j in range(i+1, n):
                    dag1, dag2 = logicxs[i].dag, logicxs[j].dag
                    upper_bound = abs(len(dag1.nodes) - len(dag2.nodes)) + abs(len(dag1.edges) - len(dag2.edges))
                    d = networkx.graph_edit_distance(dag1, dag2, node_match=lambda a, b: a['node_uuid'] == b['node_uuid'], upper_bound=2*upper_bound, timeout=2)
                    edit_distances[i, j] = edit_distances[j, i] = d if d is not None else -1
                    progress_bar.update(1)
        data_frame = pandas.DataFrame(edit_distances)
        data_frame.to_csv(output_dirpath.joinpath(f'{dataset_name}_logicx_edit_distance.csv'), index=False)


def main(input_names, input_dirpaths: list[pathlib.Path], output_dirpath: pathlib.Path, mode: Literal['junior', 'senior', 'motif', 'edit'], sample_number: int | None, worker_number: int | None):
    datasets: dict[str, list[pathlib.Path]] = {input_name: [logicx_filepath for logicx_filepath in input_dirpath.iterdir()] for input_name, input_dirpath in zip(input_names, input_dirpaths)}
    create_dir(output_dirpath)

    if mode == 'junior':
        logger.info(f'... Junior Statistics ...')
        extract_junior_statistics(datasets, output_dirpath)

    if mode== 'senior':
        logger.info(f'... Senior Statistics ...')
        extract_senior_statistics(datasets, output_dirpath)

    if mode== 'motif':
        logger.info(f'... MOTIF Occurences ...')
        extract_motif_statistics(datasets, output_dirpath)

    if mode== 'edit':
        logger.info(f'... Edit Distances ...')
        extract_edit_distances(datasets, output_dirpath)
