#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-06 09:12:21
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from younger.commons.io import load_json, save_json
from younger.commons.hash import hash_string
from younger.commons.version import semantic_release, str_to_sem

from younger_logics_ir.modules.meta import Meta
from younger_logics_ir.modules.label import Implementation
from younger_logics_ir.modules.logicx import LogicX


class Instance(object):
    _meta_filename = 'meta'
    _logicx_filename = 'logicx.bin'
    _labels_filename = 'labels.lbl'
    def __init__(self):
        self._meta: Meta = Meta(fresh_checker=self.fresh_checker)
        self._logicx: LogicX = LogicX()
        self._labels: list[Implementation] = list()

    def __hash__(self):
        return hash((self.logicx_part_unique, self.labels_part_unique))

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def valid(self) -> bool:
        return self.logicx_valid and self.labels_valid

    @property
    def unique(self) -> str | None:
        if self.valid:
            return hash_string(self.logicx_part_unique + self.labels_part_unique)
        else:
            return None

    @property
    def logicx(self) -> LogicX:
        return self._logicx

    @property
    def logicx_valid(self) -> bool:
        return self._logicx.valid

    @property
    def logicx_part_unique(self) -> str | None:
        if self.logicx_valid:
            return LogicX.luid(self._logicx)
        else:
            return None

    @property
    def labels(self) -> list[Implementation]:
        return self._labels

    @property
    def labels_valid(self) -> bool:
        return len(self._labels) != 0

    @property
    def labels_part_unique(self) -> str | None:
        if self.labels_valid:
            return '-IUID-'.join([Implementation.iuid(label) for label in self._labels])
        else:
            return None

    def fresh_checker(self) -> bool:
        return self.unique is None

    def load(self, instance_dirpath: pathlib.Path) -> None:
        if not instance_dirpath.is_dir():
            raise FileNotFoundError(f'There is no \"Instance\" can be loaded from the specified directory \"{instance_dirpath.absolute()}\".')

        meta_filepath = instance_dirpath.joinpath(self.__class__._meta_filename)
        self._load_meta(meta_filepath)
        logicx_filepath = instance_dirpath.joinpath(self.__class__._logicx_filename)
        self._load_logicx(logicx_filepath)
        labels_filepath = instance_dirpath.joinpath(self.__class__._labels_filename)
        self._load_labels(labels_filepath)
        return

    def save(self, instance_dirpath: pathlib.Path) -> None:
        if instance_dirpath.is_dir():
            raise FileExistsError(f'\"Instance\" can not be saved into the specified directory \"{instance_dirpath.absolute()}\".')

        meta_filepath = instance_dirpath.joinpath(self.__class__._meta_filename)
        self._save_meta(meta_filepath)
        logicx_filepath = instance_dirpath.joinpath(self.__class__._logicx_filename)
        self._save_logicx(logicx_filepath)
        labels_filepath = instance_dirpath.joinpath(self.__class__._labels_filename)
        self._save_labels(labels_filepath)
        return

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        if not meta_filepath.is_file():
            raise FileNotFoundError(f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".')
        self._meta.load(meta_filepath)
        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        if meta_filepath.is_file():
            raise FileExistsError(f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".')
        self._meta.save(meta_filepath)
        return

    def _load_logicx(self, logicx_filepath: pathlib.Path) -> None:
        if not logicx_filepath.is_file():
            raise FileNotFoundError(f'There is no \"LogicX\" can be loaded from the specified path \"{logicx_filepath.absolute()}\".')
        self._logicx.load(logicx_filepath)
        return

    def _save_logicx(self, logicx_filepath: pathlib.Path) -> None:
        if logicx_filepath.is_file():
            raise FileExistsError(f'\"LogicX\" can not be saved into the specified path \"{logicx_filepath.absolute()}\".')
        self._logicx.save(logicx_filepath)
        return

    def _load_labels(self, labels_filepath: pathlib.Path) -> None:
        if not labels_filepath.is_file():
            raise FileNotFoundError(f'There is no \"Lables\" can be loaded from the specified path \"{labels_filepath.absolute()}\".')
        self._labels = [Implementation.loads(s) for s in load_json(labels_filepath)]
        return

    def _save_labels(self, labels_filepath: pathlib.Path) -> None:
        if labels_filepath.is_file():
            raise FileExistsError(f'\"Labels\" can not be saved into the specified path \"{labels_filepath.absolute()}\".')
        save_json([Implementation.saves(l) for l in self._labels], labels_filepath)
        return

    def setup_logicx(self, logicx: LogicX) -> None:
        assert isinstance(logicx, LogicX), f'Argument \"logicx\" must be LogicX instead \"{type(logicx)}\"!'
        assert logicx.valid, f'Argument \"logicx\" must be valid!'
        if self.meta.is_fresh:
            self._logicx = logicx
        return

    def insert_label(self, label: Implementation) -> None:
        if self.meta.is_fresh:
            if label in self._labels:
                pass
            else:
                self._labels.append(label)
                self._labels = sorted(self._labels)
        return

    def delete_label(self, label: Implementation) -> None:
        if self.meta.is_fresh:
            if label in self._labels:
                self._labels.remove(label)
                self._labels = sorted(self._labels)
            else:
                pass
        return

    def insert(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_new()
            return True

    def delete(self) -> bool:
        if self.meta.is_fresh:
            return False
        else:
            self.meta.set_old()
            return True

    def release(self, version: semantic_release.Version) -> None:
        if self.meta.is_fresh or version == str_to_sem('0.0.0'):
            return

        if self.meta.release:
            if self.meta.is_old:
                self.meta.set_retired(version)
        else:
            if self.meta.is_new:
                self.meta.set_release(version)
        return

    @classmethod
    def copy(cls, instance: 'Instance') -> 'Instance':
        """
        Copy Instance.
        .. todo::
            This project want deepcopy and we want to check the performance of deepcopy.

        :param instance: _description_
        :type instance: Instance
        :return: _description_
        :rtype: Instance
        """
        instance_copy = Instance()
        instance_copy.setup_logicx(instance.logicx)
        for implementation in instance.labels:
            instance_copy.insert_label(Implementation.copy(implementation))

        return instance_copy

    @classmethod
    def standardize(cls, instance: 'Instance') -> tuple['Instance', list['Instance']]:
        std_logicx, std_logicx_sods = LogicX.standardize(instance.logicx)

        instance.setup_logicx(std_logicx)

        instance_sods = list() # sods: Sons or Descendants
        for std_logicx_sod in std_logicx_sods:
            std_instance = Instance()
            std_instance.setup_logicx(std_logicx_sod)
            instance_sods.append(std_instance)

        return instance, instance_sods
