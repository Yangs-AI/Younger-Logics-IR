#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-11-04 04:00:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-11-04 19:49:20
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib
import sqlite3
import datetime
import tempfile
import itertools


from younger.commons.io import load_json, save_json, saves_json, loads_json
from younger_logics_ir.commons.logging import logger

from .utils import get_model_library_name, infer_supported_frameworks


def save_model_infos(rows, save_dirpath: pathlib.Path, name: str, number_per_file: int):
    index = 0
    while True:
        model_infos = [loads_json(row[0]) for row in itertools.islice(rows, number_per_file)]
        if not model_infos:
            break
        filepath = save_dirpath.joinpath(f'{name}_{index:05d}.json')
        save_json(model_infos, filepath, indent=2)
        yield filepath.name, len(model_infos)
        index += 1


def main(model_infos_dirpath: pathlib.Path, save_dirpath: pathlib.Path, number_per_file: int, date_l: str | None = None, date_r: str | None = None) -> None:
    model_infos_filepaths = sorted(model_infos_dirpath.glob('*.json'))
    if not model_infos_filepaths:
        raise ValueError('No model info JSON files found in model_infos_dirpath.')
    if number_per_file <= 0:
        raise ValueError('number_per_file must be positive.')
    lower = datetime.datetime.strptime(date_l, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc) if date_l is not None else None
    upper = datetime.datetime.strptime(date_r, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc) if date_r is not None else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError('date_l must be earlier than date_r.')
    if any(save_dirpath.iterdir()):
        raise ValueError('Preparation requires an empty output directory to avoid stale job files.')

    minimum_date = None
    maximum_date = None

    # Keep all-model metadata and sort indexes on disk; load one input file at a time.
    with tempfile.TemporaryDirectory(prefix='.younger-conversion-prepare-', dir=save_dirpath) as temporary_dirpath:
        connection = sqlite3.connect(str(pathlib.Path(temporary_dirpath).joinpath('models.sqlite')))
        try:
            connection.execute('PRAGMA cache_size = -8192')
            connection.execute('PRAGMA temp_store = FILE')
            connection.execute('CREATE TABLE models (id TEXT PRIMARY KEY, data TEXT NOT NULL, missing_date INTEGER NOT NULL)')
            connection.execute('CREATE TABLE routes (framework TEXT, library TEXT, date TEXT, id TEXT)')
            count = 0
            for model_infos_filepath in model_infos_filepaths:
                model_infos = load_json(model_infos_filepath)
                if not isinstance(model_infos, list):
                    raise ValueError(f'Expected a model info JSON array: {model_infos_filepath}')
                for model_info in model_infos:
                    model_id = model_info['id']
                    data = saves_json(model_info)
                    existing = connection.execute('SELECT data FROM models WHERE id = ?', (model_id,)).fetchone()
                    if existing is not None:
                        if loads_json(existing[0]) != model_info:
                            raise ValueError(f'Conflicting metadata for model: {model_id}')
                        continue
                    try:
                        model_date = datetime.datetime.fromisoformat(model_info['lastModified'])
                        model_date = model_date.replace(tzinfo=datetime.timezone.utc) if model_date.tzinfo is None else model_date.astimezone(datetime.timezone.utc)
                    except (KeyError, TypeError, ValueError, AttributeError):
                        model_date = None
                    if model_date is not None:
                        minimum_date = model_date if minimum_date is None else min(minimum_date, model_date)
                        maximum_date = model_date if maximum_date is None else max(maximum_date, model_date)
                    connection.execute('INSERT INTO models VALUES (?, ?, ?)', (model_id, data, model_date is None))
                    count += 1
                    if model_date is not None and (lower is None or model_date >= lower) and (upper is None or model_date < upper):
                        for framework in sorted(infer_supported_frameworks(model_info)) or ['unknown']:
                            library = get_model_library_name(model_info) if framework == 'optimum' else None
                            connection.execute('INSERT INTO routes VALUES (?, ?, ?, ?)', (framework, library or '', model_date.isoformat(), model_id))
                    if count % 10000 == 0:
                        connection.commit()
                        logger.info(f'Indexed {count} model infos.')
                del model_infos
            connection.commit()
            connection.execute('CREATE INDEX route_order ON routes (framework, library, date, id)')
            connection.commit()

            missing_dates = connection.execute('SELECT COUNT(*) FROM models WHERE missing_date = 1').fetchone()[0]
            missing_date_files = list()
            if missing_dates:
                rows = connection.execute('SELECT data FROM models WHERE missing_date = 1 ORDER BY id')
                for filename, count in save_model_infos(rows, save_dirpath, 'missing_dates', number_per_file):
                    missing_date_files.append(filename)
            jobs = list()
            for framework, library in connection.execute('SELECT DISTINCT framework, library FROM routes ORDER BY framework, library'):
                left_name = date_l if date_l is not None else minimum_date.strftime('%Y-%m-%d')
                right_name = date_r if date_r is not None else maximum_date.strftime('%Y-%m-%d')
                date_name = f'{left_name}_{right_name}'
                group_name = f'{framework}_{library}_{date_name}' if library else f'{framework}_{date_name}'
                rows = connection.execute('SELECT models.data FROM routes JOIN models ON models.id = routes.id WHERE framework = ? AND library = ? ORDER BY routes.date, routes.id', (framework, library))
                for filename, count in save_model_infos(rows, save_dirpath, group_name, number_per_file):
                    jobs.append(
                        dict(
                            filepath=filename,
                            framework=framework,
                            library_name=library or None,
                            count=count
                        )
                    )
            save_json(
                dict(
                    date_l=date_l,
                    date_r=date_r,
                    missing_dates=missing_dates,
                    missing_date_files=missing_date_files,
                    jobs=jobs
                ),
                save_dirpath.joinpath('jobs.json'),
                indent=2
            )
            logger.info(f'Created {len(jobs)} job files; {missing_dates} models have missing or invalid dates.')
        finally:
            connection.close()
