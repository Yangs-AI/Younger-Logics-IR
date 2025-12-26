#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:54:07
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import pathlib

from typing import Generator

from younger.commons.io import load_json, save_json
from younger.commons.hash import hash_strings
from younger.commons.version import semantic_release, str_to_sem

from younger_logics_ir.modules.stamp import Stamp
from younger_logics_ir.modules.instance import Instance, LogicX

from younger_logics_ir.commons.logging import logger


class Dataset(object):
    _stamps_filename = 'stamps.json'
    _uniques_filename = 'uniques.json'
    _instances_dirname = 'instances'
    def __init__(
            self,
            instances: list[Instance] | None = None,
            version: semantic_release.Version | None = None
    ):
        instances = instances or list()
        version = version or str_to_sem('0.0.0')

        self._stamps: set[Stamp] = set()
        self._uniques: list[str] = list()
        self._instances: dict[str, Instance] = dict()

        self.insert_instances(instances)
        self.release(version)

    @property
    def uniques(self) -> list[str]:
        return self._uniques

    @property
    def instances(self) -> dict[str, Instance]:
        return self._instances

    @property
    def latest_version(self) -> semantic_release.Version:
        latest_version = str_to_sem('0.0.0')
        for stamp in self._stamps:
            latest_version = max(latest_version, stamp.version)
        return latest_version

    @property
    def checksum(self) -> str:
        exact_uniques = list()
        for unique in self._uniques:
            instance = self._instances[unique]
            if instance.meta.is_release:
                exact_uniques.append(instance.unique)
        return hash_strings(exact_uniques)

    def load(self, dataset_dirpath: pathlib.Path) -> None:
        assert dataset_dirpath.is_dir(), f'There is no \"Dataset\" can be loaded from the specified directory \"{dataset_dirpath.absolute()}\".'
        logger.info(f' = [YL-IR] = Loading Dataset @ {dataset_dirpath}...')
        stamps_filepath = dataset_dirpath.joinpath(self.__class__._stamps_filename)
        self._load_stamps(stamps_filepath)
        uniques_filepath = dataset_dirpath.joinpath(self.__class__._uniques_filename)
        self._load_uniques(uniques_filepath)
        instances_dirpath = dataset_dirpath.joinpath(self.__class__._instances_dirname)
        self._load_instances(instances_dirpath)
        return

    def save(self, dataset_dirpath: pathlib.Path) -> None:
        assert not dataset_dirpath.is_dir(), f'\"Dataset\" can not be saved into the specified directory \"{dataset_dirpath.absolute()}\".'
        logger.info(f' = [YL-IR] = Saving Dataset @ {dataset_dirpath}...')
        stamps_filepath = dataset_dirpath.joinpath(self.__class__._stamps_filename)
        self._save_stamps(stamps_filepath)
        uniques_filepath = dataset_dirpath.joinpath(self.__class__._uniques_filename)
        self._save_uniques(uniques_filepath)
        instances_dirpath = dataset_dirpath.joinpath(self.__class__._instances_dirname)
        self._save_instances(instances_dirpath)
        return

    def _load_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert stamps_filepath.is_file(), f'There is no \"Stamp\"s can be loaded from the specified path \"{stamps_filepath.absolute()}\".'
        stamps = load_json(stamps_filepath)
        self._stamps = set()
        for stamp in stamps:
            self._stamps.add(Stamp(**stamp))
        return

    def _save_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert not stamps_filepath.is_file(), f'\"Stamp\"s can not be saved into the specified path \"{stamps_filepath.absolute()}\".'
        stamps = list()
        for stamp in self._stamps:
            stamps.append(stamp.dict)
        save_json(stamps, stamps_filepath)
        return

    def _load_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert uniques_filepath.is_file(), f'There is no \"Unique\"s can be loaded from the specified path \"{uniques_filepath.absolute()}\".'
        self._uniques = load_json(uniques_filepath)
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        return

    def _save_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert not uniques_filepath.is_file(), f'\"Unique\"s can not be saved into the specified path \"{uniques_filepath.absolute()}\".'
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        save_json(self._uniques, uniques_filepath)
        return

    def _load_instances(self, instances_dirpath: pathlib.Path) -> None:
        assert instances_dirpath.is_dir(), f'There is no \"Instance\"s can be loaded from the specified directory \"{instances_dirpath.absolute()}\".'
        logger.info(f' = [YL-IR] = Loading Instances ...')
        with tqdm.tqdm(total=len(self._uniques), desc='Load Instance') as progress_bar:
            for index, unique in enumerate(self._uniques, start=1):
                progress_bar.set_description(f'Load Instance: {unique}')
                instance_dirpath = instances_dirpath.joinpath(f'{unique}')
                self._instances[unique] = Instance()
                self._instances[unique].load(instance_dirpath)
                progress_bar.update(1)
        return

    def _save_instances(self, instances_dirpath: pathlib.Path) -> None:
        assert not instances_dirpath.is_dir(), f'\"Instance\"s can not be saved into the specified directory \"{instances_dirpath.absolute()}\".'
        logger.info(f' = [YL-IR] = Saving Instances ...')
        with tqdm.tqdm(total=len(self._uniques), desc='Save Instance') as progress_bar:
            for index, unique in enumerate(self._uniques, start=1):
                progress_bar.set_description(f'Save Instance: {unique}')
                instance_dirpath = instances_dirpath.joinpath(f'{unique}')
                instance = self._instances[unique]
                instance.save(instance_dirpath)
                progress_bar.update(1)
        return

    def insert_instance(self, instance: Instance) -> bool:
        # Insert only instances that do not belong to any dataset.
        assert isinstance(instance, Instance), f'Argument \"instance\"must be an \"Instance\" instead \"{type(instance)}\"!'
        if instance.unique is None:
            return False
        else:
            if instance.unique in self._instances:
                return False
            else:
                self._instances[instance.unique] = instance
                return self._instances[instance.unique].insert()

    def delete_instance(self, instance: Instance) -> bool:
        # Delete only the instances within the dataset.
        assert isinstance(instance, Instance), f'Argument \"instance\"must be an \"Instance\" instead \"{type(instance)}\"!'
        if instance.unique is None:
            return False
        else:
            if instance.unique in self._instances:
                instance = self._instances[instance.unique]
                return instance.delete()
            else:
                return False

    def insert_instances(self, instances: list[Instance]) -> int:
        flags = list()
        for instance in instances:
            flags.append(self.insert_instance(instance))
        return sum(flags)

    def delete_instances(self, instances: list[Instance]) -> int:
        flags = list()
        for instance in instances:
            flags.append(self.delete_instance(instance))
        return sum(flags)

    def release(self, version: semantic_release.Version) -> None:
        if version == str_to_sem('0.0.0'):
            return
        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}\n'
            f'Latest: {self.latest_version}'
        )

        for unique, instance in self._instances.items():
            if instance.meta.is_external:
                if instance.meta.is_new:
                    self._uniques.append(unique)
                if instance.meta.is_old:
                    self._instances.pop(unique)
            instance.release(version)

        stamp = Stamp(
            str(version),
            self.checksum,
        )
        if stamp in self._stamps:
            return
        else:
            self._stamps.add(stamp)
        return

    def acquire(self, version: semantic_release.Version) -> 'Dataset':
        logger.info(f' = [YL-IR] = Acquiring Dataset: version = {version}...')
        dataset = Dataset()
        with tqdm.tqdm(total=len(self._uniques), desc='Acquire Instance') as progress_bar:
            for index, unique in enumerate(self._uniques, start=1):
                instance = self._instances[unique]
                if (instance.meta.release and instance.meta.release_version <= version) and (not instance.meta.retired or version < instance.meta.retired_version):
                    progress_bar.set
                    dataset.insert_instance(instance)
        dataset.release(version=version)
        return dataset

    def check(self) -> None:
        assert len(self._uniques) == len(self._instances), f'The number of \"Instance\"s does not match the number of \"Unique\"s.'
        for unique in self._uniques:
            instance = self._instances[unique]
            assert unique == instance.unique, f'The \"Unique={instance.unique}\" of \"Instance\" does not match \"Unique={unique}\" '
        return

    def clean(self) -> None:
        """
        Warn! This method will initialize all the properties of the dataset (including the stamps, uniques, and instances).
        """

        self._stamps: set[Stamp] = set()
        self._uniques: list[str] = list()
        self._instances: dict[str, Instance] = dict()

    @classmethod
    def drain_instances(cls, instances_dirpath: pathlib.Path, strict: bool = False) -> Generator[Instance, None, None]:
        """

        :param instances_dirpath: _description_
        :type instances_dirpath: pathlib.Path
        :param strict: _description_, defaults to False
        :type strict: bool, optional

        :raises exception: _description_

        :return: _description_
        :rtype: list[Instance]
        """

        # instances = list()
        # logger.info(f' = [YL-IR] = Draining Instances @ {instances_dirpath}...')
        instance_dirpaths = sorted(instances_dirpath.iterdir())
        # with tqdm.tqdm(total=len(instance_dirpaths), desc='Drain Instance') as progress_bar:
        for index, instance_dirpath in enumerate(instance_dirpaths, start=1):
            instance = Instance()
            try:
                # progress_bar.set_description(f'Drain Instance[S]: {instance_dirpath.name}')
                instance.load(instance_dirpath)
            except Exception as exception:
                # progress_bar.set_description(f'Drain Instance[F]: {instance_dirpath.name}')
                if strict:
                    raise exception
                else:
                    continue
            # instances.append(instance)
            yield instance
            # progress_bar.update(1)

        # return instances

    @classmethod
    def flush_instances(cls, instances: list[Instance], instances_dirpath: pathlib.Path, strict: bool = False) -> None:
        """

        :param instances: _description_
        :type instances: list[Instance]
        :param instances_dirpath: _description_
        :type instances_dirpath: pathlib.Path
        :param strict: _description_, defaults to False
        :type strict: bool, optional

        :raises exception: _description_
        """

        logger.info(f' = [YL-IR] = Flushing Instances @ {instances_dirpath}...')
        with tqdm.tqdm(total=len(instances), desc='Flush Instance') as progress_bar:
            for index, instance in enumerate(instances, start=1):
                instance_unique = instance.unique
                instance_dirpath = instances_dirpath.joinpath(f'{instance_unique}')
                try:
                    progress_bar.set_description(f'Flush Instance[S]: {instance_unique}')
                    instance.save(instance_dirpath)
                except Exception as exception:
                    progress_bar.set_description(f'Flush Instance[F]: {instance_unique}')
                    if strict:
                        raise exception
                    else:
                        continue

                progress_bar.update(1)

    @classmethod
    def drain_logicxs(cls, logicxs_dirpath: pathlib.Path, strict: bool = False) -> Generator[LogicX, None, None]:
        """

        :param logicxs_dirpath: _description_
        :type logicxs_dirpath: pathlib.Path
        :param strict: _description_, defaults to False
        :type strict: bool, optional

        :raises exception: _description_

        :return: _description_
        :rtype: list[LogicX]
        """

        # logicxs = list()
        # logger.info(f' = [YL-IR] = Draining LogicXs @ {logicxs_dirpath}...')
        logicx_filepaths = sorted(logicxs_dirpath.iterdir())
        # with tqdm.tqdm(total=len(logicx_dirpaths), desc='Drain Instance') as progress_bar:
        for index, logicx_filepath in enumerate(logicx_filepaths, start=1):
            logicx = LogicX()
            try:
                # progress_bar.set_description(f'Drain LogicX[S]: {logicx_dirpath.name}')
                logicx.load(logicx_filepath)
            except Exception as exception:
                # progress_bar.set_description(f'Drain LogicX[F]: {logicx_dirpath.name}')
                if strict:
                    raise exception
                else:
                    continue
            # logicxs.append(logicx)
            yield logicx
            # progress_bar.update(1)

        # return logicxs

    @classmethod
    def flush_logicxs(cls, logicxs: list[Instance], logicxs_dirpath: pathlib.Path, strict: bool = False) -> None:

        logger.info(f' = [YL-IR] = Flushing LogicX @ {logicxs_dirpath}...')
        with tqdm.tqdm(total=len(logicxs), desc='Flush LogicX') as progress_bar:
            for index, logicx in enumerate(logicxs, start=1):
                logicx_unique = LogicX.hash(logicx)
                logicx_filepath = logicxs_dirpath.joinpath(f'{logicx_unique}')
                try:
                    progress_bar.set_description(f'Flush LogicX[S]: {logicx_unique}')
                    logicx.save(logicx_filepath)
                except Exception as exception:
                    progress_bar.set_description(f'Flush LogicX[F]: {logicx_unique}')
                    if strict:
                        raise exception
                    else:
                        continue

                progress_bar.update(1)
