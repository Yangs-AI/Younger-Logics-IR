#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-16 10:31:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-05-01 19:18:51
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Any

from younger.commons.io import loads_json, saves_json, get_object_with_sorted_dict
from younger.commons.hash import hash_string


class Origin(object):
    def __init__(self, hub: str, owner: str, name: str, artifact: str | None = None):
        """

        :param hub: _description_
        :type hub: str
        :param owner: _description_
        :type owner: str
        :param name: _description_
        :type name: str
        :param artifact: _description_
        :type artifact: str

        Hub: The hub where the Implementation is located
        Owner: The owner of the Implementation
        Name: The name of the Implementation
        Artifact: The artifact (remote filepath) of the Implementation
        <Hub, Owner, Name, Artifact> is the unique identifier of the Implementation

        """

        self._hub = hub
        self._owner = owner
        self._name = name
        self._artifact = artifact

    def __hash__(self):
        return hash((self.hub, self.owner, self.name, self.artifact))

    def __str__(self):
        return f'Origin - <Hub/Owner/Name/Artifact>: <{self.hub}/{self.owner}/{self.name}/{self.artifact}>'

    def __eq__(self, other: 'Origin') -> bool:
        return self.hub == other.hub and self.owner == other.owner and self.name == other.name and self.artifact == other.artifact

    @property
    def hub(self) -> str:
        return self._hub

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def name(self) -> str:
        return self._name

    @property
    def artifact(self) -> str | None:
        return self._artifact

    @classmethod
    def load_dict(cls, d: dict) -> 'Origin':
        o = Origin(
            d['hub'],
            d['owner'],
            d['name'],
            d['artifact'],
        )
        return o

    @classmethod
    def save_dict(cls, o: 'Origin') -> dict:
        d = dict(
            hub=o._hub,
            owner=o._owner,
            name=o._name,
            artifact=o._artifact,
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Origin':
        d = loads_json(t)
        o = cls.load_dict(d)
        return o

    @classmethod
    def saves(cls, o: 'Origin') -> str:
        d = cls.save_dict(o)
        t = saves_json(d)
        return t


class Benchmark(object):
    def __init__(self, task: str, dataset: str):
        """

        :param task: _description_
        :type task: str
        :param dataset: Training Dataset
        :type dataset: str

        Task: The task of the Benchmark
        Dataset: The training dataset of the Benchmark
        <Task, Dataset> is the unique identifier of the Benchmark

        """

        self._task = task
        self._dataset = dataset

    def __hash__(self):
        return hash((self.task, self.dataset))

    def __str__(self):
        return f'Benchmark - <Task/Dataset>: <{self.task}/{self.dataset}>'

    def __eq__(self, other: 'Benchmark') -> bool:
        return self.task == other.task and self.dataset == other.dataset

    @property
    def task(self) -> str:
        return self._task

    @property
    def dataset(self) -> str:
        return self._dataset

    @classmethod
    def load_dict(cls, d: dict) -> 'Benchmark':
        b = Benchmark(
            d['task'],
            d['dataset'],
        )
        return b

    @classmethod
    def save_dict(cls, b: 'Benchmark') -> dict:
        d = dict(
            task=b._task,
            dataset=b._dataset,
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Benchmark':
        d = loads_json(t)
        b = cls.load_dict(d)
        return b

    @classmethod
    def saves(cls, b: 'Benchmark') -> str:
        d = cls.save_dict(b)
        t = saves_json(d)
        return t


class Evaluation(object):
    def __init__(self, dataset: str, metric: str):
        """

        :param dataset: Evaluation set (e.g., validation set, test set (coco test-dev), etc.)
        :type dataset: str
        :param metric: _description_
        :type metric: str

        """

        self._dataset = dataset
        self._metric = metric

    def __hash__(self):
        return hash((self.dataset, self.metric))

    def __str__(self):
        return f'Evaluation - <Dataset/Metric>: <{self.dataset}/{self.metric}>'

    def __eq__(self, other: 'Evaluation') -> bool:
        return self.dataset == other.dataset and self.metric == other.metric

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def metric(self) -> str:
        return self._metric

    @classmethod
    def load_dict(cls, d: dict) -> 'Evaluation':
        e = Evaluation(
            d['dataset'],
            d['metric'],
        )
        return e

    @classmethod
    def save_dict(cls, e: 'Evaluation') -> dict:
        d = dict(
            dataset=e._dataset,
            metric=e._metric,
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Evaluation':
        d = loads_json(t)
        e = cls.load_dict(d)
        return e

    @classmethod
    def saves(cls, e: 'Evaluation') -> str:
        d = cls.save_dict(e)
        t = saves_json(d)
        return t


class Implementation(object):
    def __init__(self, origin: Origin, like: int | None = None, download: int | None = None, performances: dict[Benchmark, dict[Evaluation, Any]] | None = None):
        """

        :param origin: _description_
        :type origin: Origin
        :param like: _description_
        :type like: int | None
        :param download: _description_
        :type download: int | None
        :param performances: _description_
        :type performances: dict[Benchmark, dict[Evaluation, Any]] | None

        Below are the properties of the Implementation
        Like: The number of likes of the Implementation, it can be used to evaluate the popularity of the Implementation (Int). Some Implementation does not have the like number (None).
        Download: The number of downloads of the Implementation, it can be used to evaluate the popularity of the Implementation (Int). Some Implementation does not have the download number (None).
        Performances: The performances of the Implementation, it is a dictionary with the key as the Benchmark and the value as the Evaluations on the Benchmark. Some Implementation does not provide the performances (None). The Evaluations is a dictionary with the key as the Evaluation and the value as the Performance value (Any).

        <Benchmark, Evaluation> is the unique identifier of a specific Performance. The Performance is the Evaluation of the Implementation on the Benchmark.

        """

        self._origin = origin

        self._like = None
        self._download = None
        self._performances = None

        self.setup_like(like)
        self.setup_download(download)
        self.setup_performances(performances)

    def __hash__(self):
        return hash(self.origin)

    def __str__(self):
        return f'Implementation - <Origin/Like/Download/#Performances>: <{self.origin}/{self.like}/{self.download}/{len(self.count_performances())}>'

    def __eq__(self, other: 'Implementation') -> bool:
        return self.origin == other.origin

    @property
    def origin(self) -> Origin:
        return self._origin

    @property
    def like(self) -> int | None:
        return self._like

    @property
    def download(self) -> int | None:
        return self._download

    @property
    def performances(self) -> dict[Benchmark, dict[Evaluation, Any]] | None:
        return self._performances

    def setup_like(self, like: int | None) -> None:
        self._like = like

    def setup_download(self, download: int | None) -> None:
        self._download = download

    def setup_performances(self, performances: dict[Benchmark, dict[Evaluation, Any]] | None) -> None:
        self._performances = dict()
        performances = performances or dict()
        for benchmark, evaluation2performance in performances.items():
            for evaluation, performance in evaluation2performance.items():
                self.insert_performance(benchmark, evaluation, performance)

    @classmethod
    def load_dict(cls, d: dict) -> 'Implementation':
        i = Implementation(
            Origin.load_dict(d['origin']),
            d['like'],
            d['download'],
            get_object_with_sorted_dict({
                Benchmark.load_dict(benchmark): {
                    Evaluation.load_dict(evaluation): performance
                    for evaluation, performance in evaluation2performance.items()
                }
                for benchmark, evaluation2performance in d['performances'].items() # {benchmark: {evaluation: performance}}
            })
        )
        return i

    @classmethod
    def save_dict(cls, i: 'Implementation') -> dict:
        d = dict(
            origin=Origin.save_dict(i._origin),
            like=i._like,
            download=i._download,
            performances=get_object_with_sorted_dict({
                Benchmark.saves(benchmark): {
                    Evaluation.saves(evaluation): performance
                    for evaluation, performance in evaluation2performance.items()
                }
                for benchmark, evaluation2performance in i._performances.items()
            })
        )
        return d

    @classmethod
    def loads(cls, t: str) -> 'Implementation':
        d = loads_json(t)
        i = cls.load_dict(d)
        return i

    @classmethod
    def saves(cls, i: 'Implementation') -> str:
        d = cls.save_dict(i)
        t = saves_json(d)
        return t

    @classmethod
    def hash(cls, implementation: 'Implementation') -> str:
        """
        Implementation Hash (Hash)
        Only the Origin is used to calculate the hash of the Implementation.
        Because an Implementation is uniquely identified by its Origin.

        .. note::
            This classmethod is different from the __hash__ method of the Implementation class which returns the int type hash value of the Implementation object for the convinent of quickly checking the equality of two Implementation objects.

        :param implementation: _description_
        :type implementation: Implementation

        :return: _description_
        :rtype: str
        """
        return hash_string(Origin.saves(implementation.origin))

    @classmethod
    def iuid(cls, implementation: 'Implementation') -> str:
        """
        Implementation Unique ID (IUID)

        :param implementation: _description_
        :type implementation: Implementation

        :return: _description_
        :rtype: str
        """
        return hash_string(cls.saves(implementation))

    @classmethod
    def copy(cls, implementation: 'Implementation') -> 'Implementation':
        """
        Copy Implementation.
        .. todo::
            This project want deepcopy and we want to check the performance of deepcopy.

        :param implementation: _description_
        :type implementation: Implementation
        :return: _description_
        :rtype: Implementation
        """
        return Implementation(
            implementation.origin,
            implementation.like,
            implementation.download,
            implementation.performances
        )

    def search_performance(self, benchmark: Benchmark, evaluation: Evaluation) -> Any:
        return self._performances.get(benchmark, dict()).get(evaluation, None)

    def insert_performance(self, benchmark: Benchmark, evaluation: Evaluation, performance: Any) -> None:
        evaluation2performance = self._performances.setdefault(benchmark, dict())
        if evaluation not in evaluation2performance:
            evaluation2performance[evaluation] = performance

    def update_performance(self, benchmark: Benchmark, evaluation: Evaluation, performance: Any) -> None:
        evaluation2performance = self._performances.get(benchmark, dict())
        if evaluation in evaluation2performance:
            evaluation2performance[evaluation] = performance
        else:
            pass

    def delete_performance(self, benchmark: Benchmark, evaluation: Evaluation) -> Any:
        evaluation2performance = self._performances.get(benchmark, dict())
        if evaluation in evaluation2performance:
            evaluation2performance.pop(evaluation)
        else:
            pass

    def count_performances(self, benchmark: Benchmark | None, evaluation: Evaluation | None) -> int:
        if benchmark is None:
            if evaluation is None:
                return sum([len(evaluation2performance) for evaluation2performance in self._performances.values()])
            else:
                return sum([1 if evaluation in evaluation2performance else 0 for evaluation2performance in self._performances.values()])
        else:
            if evaluation is None:
                return len(self._performances.get(benchmark, dict()))
            else:
                return 1 if evaluation in self._performances.get(benchmark, dict()) else 0
