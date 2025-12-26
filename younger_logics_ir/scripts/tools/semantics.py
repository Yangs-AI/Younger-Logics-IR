#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-09 21:42:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:58:30
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import numpy
import pandas
import hdbscan
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Literal
from sklearn.metrics.pairwise import cosine_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from younger.commons.io import create_dir

from younger_logics_ir.commons.logging import logger


def extract_embedding_cos(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    centers = {k: v.mean(axis=0) for k, v in datasets.items()}

    names = list(datasets.keys())
    cos_matrix = pandas.DataFrame(index=names, columns=names)
    for a in names:
        for b in names:
            cos_matrix.loc[a, b] = cosine_distances(centers[a].reshape(1, -1), centers[b].reshape(1, -1))[0][0]
    cos_matrix = cos_matrix.astype(float)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cos_matrix, annot=True, cmap='Blues')
    # plt.title('Cosine Distance Between Dataset Embedding Centers')
    plt.tight_layout()
    plt.savefig(output_dirpath.joinpath('embedding_cos_heatmap.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    cos_matrix.to_csv(output_dirpath.joinpath('embedding_cos_matrix.csv'))


def compute_mmd(x: numpy.ndarray, y: numpy.ndarray, gamma=1.0, kernel: Literal['rbf'] = 'rbf'):
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def extract_embedding_mmd(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    names = list(datasets.keys())
    mmd_matrix = pandas.DataFrame(index=names, columns=names)
    for a in names:
        for b in names:
            mmd_matrix.loc[a, b] = compute_mmd(datasets[a], datasets[b])
    mmd_matrix = mmd_matrix.astype(float)

    plt.figure(figsize=(6, 5))
    sns.heatmap(mmd_matrix, annot=True, cmap='coolwarm')
    # plt.title('MMD Distance Between Dataset Embeddings')
    plt.tight_layout()
    plt.savefig(output_dirpath.joinpath('embedding_mmd_heatmap.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    mmd_matrix.to_csv(output_dirpath.joinpath('embedding_mmd_matrix.csv'))


def extract_embedding_cls(datasets: dict[str, numpy.ndarray], output_dirpath: pathlib.Path):
    reducers = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, init='pca', learning_rate='auto'),
        'umap': UMAP(n_components=2)
    }

    summary = []
    for name, data in datasets.items():
        for method, reducer in reducers.items():
            reduced = reducer.fit_transform(data)
            reduced = MinMaxScaler().fit_transform(reduced)

            # ====== KMeans ======
            best_k = 2
            best_score = -1
            for k in range(2, min(11, len(reduced))):
                kmeans = KMeans(n_clusters=k, n_init='auto').fit(reduced)
                score = silhouette_score(reduced, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k

            final_kmeans = KMeans(n_clusters=best_k, n_init='auto').fit(reduced)
            df_kmeans = pandas.DataFrame(reduced, columns=['x', 'y'])
            df_kmeans['label'] = final_kmeans.labels_
            plt.figure(figsize=(5, 4))
            sns.scatterplot(x='x', y='y', hue='label', data=df_kmeans, palette='tab10', s=10)
            # plt.title(f'{name} - {method.upper()} + KMeans (k={best_k}, Silhouette={best_score:.2f})')
            plt.legend().remove()
            plt.tight_layout()
            filename_kmeans = output_dirpath.joinpath(f'{name}_embedding_{method}_kmeans.pdf')
            plt.savefig(filename_kmeans, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()

            summary.append({
                'dataset': name,
                'reducer': method,
                'method': 'kmeans',
                'n_clusters': best_k,
                'silhouette_score': best_score,
            })

            # ====== HDBSCAN ======
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                score = silhouette_score(reduced[labels != -1], labels[labels != -1])
            else:
                score = -1

            df_hdb = pandas.DataFrame(reduced, columns=['x', 'y'])
            df_hdb['label'] = labels
            plt.figure(figsize=(5, 4))
            sns.scatterplot(x='x', y='y', hue='label', data=df_hdb, palette='tab10', s=10)
            # plt.title(f'{name} - {method.upper()} + HDBSCAN (k={n_clusters}, Silhouette={score:.2f})')
            plt.legend().remove()
            plt.tight_layout()
            filename_hdbscan = output_dirpath.joinpath(f'{name}_embedding_{method}_hdbscan.pdf')
            plt.savefig(filename_hdbscan, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()

            summary.append({
                'dataset': name,
                'reducer': method,
                'method': 'hdbscan',
                'n_clusters': n_clusters,
                'silhouette_score': score,
            })

    pandas.DataFrame(summary).to_csv(output_dirpath.joinpath('embedding_cls_summary.csv'), index=False)


def main(input_names: list[str], input_filepaths: list[pathlib.Path], output_dirpath: pathlib.Path, mode: Literal['cos', 'mmd', 'cls'], standardize: bool = False):
    datasets: dict[str, list[pathlib.Path]] = {input_name: pandas.read_csv(input_filepath).to_numpy() for input_name, input_filepath in zip(input_names, input_filepaths) }
    create_dir(output_dirpath)

    if standardize:
        scaler = StandardScaler()
        datasets = {k: scaler.fit_transform(v) for k, v in datasets.items()}

    if mode == 'cos':
        logger.info(f'... COS Statistics ...')
        extract_embedding_cos(datasets, output_dirpath)

    if mode == 'mmd':
        logger.info(f'... MMD Statistics ...')
        extract_embedding_mmd(datasets, output_dirpath)

    if mode == 'cls':
        logger.info(f'... CLS Statistics ...')
        extract_embedding_cls(datasets, output_dirpath)
