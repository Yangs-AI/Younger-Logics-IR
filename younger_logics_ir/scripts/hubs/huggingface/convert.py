#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-11-04 20:29:01
# Copyright (c) 2024 Yangs.AI
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import sys
import contextlib
import onnx
import tqdm
import pathlib
import multiprocessing

from typing import Any, Literal, Callable

from huggingface_hub import login, hf_hub_download, snapshot_download, utils

from younger.commons.io import saves_json, loads_json, save_json, load_json, create_dir, delete_dir, get_human_readable_size_representation
from younger.commons.hash import hash_string

from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert
from younger_logics_ir.converters.onnx2ir.io import load_model, check_model

from younger_logics_ir.commons.logging import logger
from younger_logics_ir.commons.constants import YLIROriginHub

from younger_logics_ir.scripts.commons.utils import get_onnx_opset_versions, get_onnx_model_opset_version

from .http import RateLimitConfig, configure_hub_http, export_process

from .utils import get_model_library_name, get_huggingface_hub_model_readme, get_huggingface_hub_model_siblings, clean_huggingface_hub_model_cache, infer_supported_frameworks, is_permanent_error, get_minimum_opset_from_error, get_huggingface_hub_model_storage


def clean_cache(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path):
    delete_dir(cvt_cache_dirpath, only_clean=True)
    clean_huggingface_hub_model_cache(model_id, ofc_cache_dirpath)


@contextlib.contextmanager
def export_output(verbose: bool):
    saved_fds = {}
    try:
        if not verbose:
            sys.stdout.flush()
            sys.stderr.flush()
            with open(os.devnull, 'w') as devnull:
                for fd in (1, 2):
                    saved_fds[fd] = os.dup(fd)
                    os.dup2(devnull.fileno(), fd)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            for fd, saved in saved_fds.items():
                try:
                    os.dup2(saved, fd)
                finally:
                    os.close(saved)


def safe_optimum_export(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, device: str, library_name: str | None = None, verbose: bool = False):
    import os
    import inspect

    # Keep worker output visible when debugging; otherwise suppress it before
    # importing optimum/torch. Results always return through results_queue.
    saved_fds = {}
    devnull = None

    try:
        if not verbose:
            # Best-effort stdout/stderr redirection.
            try:
                for fd in (1, 2):
                    saved_fds[fd] = os.dup(fd)

                devnull = os.open(os.devnull, os.O_WRONLY)

                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
            except OSError:
                # Redirection failure should not prevent model export.
                pass
            finally:
                if devnull is not None:
                    try:
                        os.close(devnull)
                    except OSError:
                        pass

        try:
            from optimum.exporters.onnx import main_export

            export_kwargs = dict(
                opset=onnx_opset_version,
                device=device,
                cache_dir=ofc_cache_dirpath,
                monolith=True,
                do_validation=False,
                trust_remote_code=True,
                no_post_process=True,
                library_name=library_name,
            )

            # Why switch by opset:
            # - Optimum docs recommend dynamo exporter for opset >= 18, while
            #   opset < 18 can keep the legacy TorchScript exporter path.
            # - TorchScript is part of PyTorch; no standalone TorchScript package
            #   is required.
            # Ref: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
            # Keep a guard for environments where main_export has no `dynamo`
            # parameter (older optimum versions).
            if 'dynamo' in inspect.signature(main_export).parameters:
                export_kwargs['dynamo'] = (onnx_opset_version >= 18)

            # Why not enable custom export knobs here by default:
            # The custom path (model_kwargs/custom_onnx_configs/fn_get_submodels)
            # is model-family-specific and may require per-architecture config.
            # The current pipeline favors broad, stable batch conversion.
            # Ref: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model#customize-the-export-of-official-transformers-models

            main_export(model_id, cvt_cache_dirpath, **export_kwargs)
            this_status = 'success'
            this_error = ''
        except MemoryError as exception:
            if verbose:
                logger.exception(f'Fail to export Optimum model: {model_id}; opset: {onnx_opset_version}')
            this_status = 'oversize'
            this_error = str(exception)
        except utils.RepositoryNotFoundError as exception:
            if verbose:
                logger.exception(f'Fail to export Optimum model: {model_id}; opset: {onnx_opset_version}')
            this_status = 'access_deny'
            this_error = str(exception)
        except Exception as exception:
            if verbose:
                logger.exception(f'Fail to export Optimum model: {model_id}; opset: {onnx_opset_version}')
            error_text = f"{type(exception).__name__}: {exception}"

            if (
                type(exception).__name__ == "OutOfMemoryError"
                or "CUDA out of memory" in error_text
                or "out of memory" in error_text.lower()
            ):
                this_status = "oversize"
            else:
                this_status = "convert_error"

            this_error = error_text
    except Exception as exception:
        if verbose:
            logger.exception(f'Fail to export Optimum model: {model_id}; opset: {onnx_opset_version}')
        this_status = "covert_worker_error"
        this_error = f"{type(exception).__name__}: {exception}"

    finally:
        # Restore all saved descriptors, including partially redirected ones.
        for fd, saved in saved_fds.items():
            try:
                os.dup2(saved, fd)
            except OSError:
                pass
            finally:
                try:
                    os.close(saved)
                except OSError:
                    pass

        results_queue.put((this_status, this_error))


def convert_optimum(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None, verbose: bool = False) -> tuple[dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]], list[Instance], list[tuple[str, int]]]:
    assert device in {'cpu', 'cuda'}
    status: dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]] = dict()
    instances: list[Instance] = list()
    artifacts: list[tuple[str, int]] = list()

    # The highest opset version supported by torch.onnx.export is 20
    # torch.onnx.export & optimum.onnx.main_export decide the highest supported opset version.
    # TODO: It is a complex decision to determine the highest supported opset version accurately.
    # TODO: Please make this decision more robust and accurate.
    from torch.onnx import _constants
    opset_versions = [v for v in get_onnx_opset_versions() if v <= _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET]
    opset_index = 0
    while opset_index < len(opset_versions):
        onnx_opset_version = opset_versions[opset_index]
        logger.info(f'[{model_id}] [opset {onnx_opset_version}] Starting Optimum export; library: {library_name}; device: {device}')
        delete_dir(cvt_cache_dirpath, only_clean=True)
        results_queue = multiprocessing.Queue()
        subprocess = export_process(target=safe_optimum_export, args=(model_id, cvt_cache_dirpath, ofc_cache_dirpath, onnx_opset_version, results_queue, device, library_name, verbose))
        subprocess.start()
        subprocess.join()

        this_status_details: dict[str, Literal['success', 'logicx_error']] = dict()
        if results_queue.empty():
            this_status = 'system_kill'
            this_error = 'subprocess exited without putting result (likely OOM)'
        else:
            this_status, this_error = results_queue.get()
            if this_status == 'success':
                logger.info(f'[{model_id}] [opset {onnx_opset_version}] Optimum export completed; converting ONNX artifacts to IR')
                for filepath in cvt_cache_dirpath.rglob('*.onnx'):
                    artifact = filepath.relative_to(cvt_cache_dirpath).as_posix()
                    logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {artifact}')
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(filepath)))
                        instances.append(instance)
                        artifacts.append((artifact, onnx_opset_version))
                        this_status_details[artifact] = 'success'
                    except Exception as exception:
                        logger.error(f'Fail in convert_optimum: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                        this_status_details[artifact] = 'logicx_error'

        if this_status == 'access_deny':
            # jump out of the opset loop if Repository can not found
            logger.warning(f'[{model_id}] [opset {onnx_opset_version}] {this_status}: {this_error}')
            status[onnx_opset_version] = (this_status, this_status_details)
            # A 404 is a repository-level error rather than an opset-specific failure.
            # Stop here, so `status` may not contain entries for subsequent opsets.
            # Keep this semantic difference in mind if this logic is changed in the future.
            break

        if this_status == 'convert_error':
            # Permanent error -> skip remaining opsets entirely
            if is_permanent_error(this_error):
                logger.warning(f'[{model_id}] [opset {onnx_opset_version}] {this_status}: {this_error}')
                status[onnx_opset_version] = (this_status, this_status_details)
                break

            # Unsupported operator with known minimum version -> jump there
            target_opset = get_minimum_opset_from_error(this_error)
            if target_opset is not None and target_opset > onnx_opset_version:
                logger.warning(f'[{model_id}] [opset {onnx_opset_version}] {this_status}: {this_error}')
                status[onnx_opset_version] = (this_status, this_status_details)
                # Find the opset >= target
                for i in range(opset_index + 1, len(opset_versions)):
                    if opset_versions[i] >= target_opset:
                        opset_index = i
                        break
                else:
                    # target_opset beyond our range -> exit
                    break
                continue

        # Log and record status
        if this_status == 'success':
            logger.info(f'[{model_id}] [opset {onnx_opset_version}] {this_status}: {len(this_status_details)} artifacts created')
        elif this_error:
            logger.warning(f'[{model_id}] [opset {onnx_opset_version}] {this_status}: {this_error}')
        status[onnx_opset_version] = (this_status, this_status_details)
        opset_index += 1

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_sb3_export(model_id: str, sb3_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, verbose: bool = False):
    try:
        with export_output(verbose):
            import torch
            import gymnasium
            import stable_baselines3

            sb3_algorithm_names = set(['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'])
            def infer_sb3_algorithm(model_id: str, sb3_model_path: pathlib.Path) -> Literal['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'] | None:
                # First, try to infer the algorithm from the model file name.
                for sb3_algorithm_name in sb3_algorithm_names:
                    if sb3_algorithm_name in sb3_model_path.name.lower():
                        return sb3_algorithm_name

                # If that fails, try to infer the algorithm from the model ID.
                for sb3_algorithm_name in sb3_algorithm_names:
                    if sb3_algorithm_name in model_id.lower():
                        return sb3_algorithm_name

                return None

            def load_sb3_model(sb3_model_path: pathlib.Path, sb3_algorithm_name: Literal['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'] | None):
                if sb3_algorithm_name is None:
                    for sb3_algorithm_name in sb3_algorithm_names:
                        try:
                            return stable_baselines3.__dict__[sb3_algorithm_name.upper()].load(sb3_model_path, device='cpu', custom_objects={"optimize_memory_usage": False, "handle_timeout_termination": False})
                        except Exception:
                            continue
                    raise ValueError(f"Current model at {sb3_model_path} could not be loaded because the algorithm could not be inferred, this may indicate an unsupported algorithm.")
                else:
                    return stable_baselines3.__dict__[sb3_algorithm_name.upper()].load(sb3_model_path, custom_objects={"optimize_memory_usage": False, "handle_timeout_termination": False})

            class PureMathWrapper(torch.nn.Module):
                #Bypasses the strict probability distribution check in PyTorch 2.0+ Dynamo
                #by manually mapping the pure mathematical tensor flow.
                def __init__(self, model):
                    super().__init__()
                    self.model_name = model.__class__.__name__
                    self.action_space = model.action_space
                    self.observation_space = model.observation_space

                    # Security interceptor: Block RNNs and Mask mechanisms to prevent ONNX graph explosion
                    # Keep extraction conservative. RNN/maskable policies often need
                    # additional state inputs that this generic exporter does not support.
                    if 'recurrent' in self.model_name.lower() or 'maskable' in self.model_name.lower():
                        raise NotImplementedError(f'Unsupported SB3 family for export: {self.model_name}.')

                    if self.model_name.upper() in {'SAC', 'TD3', 'DDPG', 'TQC'}:
                        self.net = model.policy.actor
                    elif self.model_name.upper() in {'DQN', 'QRDQN'}:
                        self.net = model.policy.q_net
                    else:
                        # Covers PPO, A2C, TRPO, etc.
                        self.net = model.policy

                    self.is_dict = isinstance(model.observation_space, gymnasium.spaces.Dict)
                    if self.is_dict:
                        self.keys = list(model.observation_space.spaces.keys())
                    else:
                        self.keys = list()

                def forward(self, *args):
                    if self.is_dict:
                        observations = {k: v for k, v in zip(self.keys, args)}
                    else:
                        observations = args[0]

                    if self.model_name.upper() in {'DQN', 'QRDQN', 'TD3', 'DDPG'}:
                        return self.net(observations)

                    if self.model_name.upper() in {'SAC', 'TQC'}:
                        features = self.net.extract_features(observations, self.net.features_extractor)
                        latent_pi = self.net.latent_pi(features)
                        return torch.tanh(self.net.mu(latent_pi))

                    features = self.net.extract_features(observations)
                    if self.net.share_features_extractor:
                        latent_pi, _ = self.net.mlp_extractor(features)
                    else:
                        latent_pi = self.net.mlp_extractor.forward_actor(features[0])

                    mean_actions = self.net.action_net(latent_pi)
                    if isinstance(self.action_space, gymnasium.spaces.Discrete):
                        return torch.argmax(mean_actions, dim=1)
                    return mean_actions

            sb3_algorithm_name = infer_sb3_algorithm(model_id, sb3_model_path)
            model = load_sb3_model(sb3_model_path, sb3_algorithm_name)
            model= PureMathWrapper(model)
            model.eval()

            model_device = next(model.parameters()).device
            if model.is_dict:
                dummy_input = tuple(torch.randn(1, *model.observation_space.spaces[key].shape).to(model_device) for key in model.keys)
                input_names = [f'input_{key}' for key in model.keys]
            else:
                dummy_input = (torch.randn(1, *model.observation_space.shape).to(model_device), )
                input_names = ['input_observation']

            torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=onnx_opset_version, input_names=input_names, output_names=['output_action'])
            this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'
        logger.error(f'Fail to export SB3 model: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
    finally:
        results_queue.put(this_status)


def convert_sb3(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None, verbose: bool = False) -> tuple[dict[str, dict[int, Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[tuple[str, int]]]:
    status: dict[str, dict[int, Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[tuple[str, int]] = list()

    logger.info(f'[{model_id}] Listing sb3 model files')
    remote_sb3_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.zip'])
    # The lowest opset version support for stable-baselines3 models is 14
    # torch.onnx.export & optimum.onnx.main_export decide the highest supported opset version.
    # TODO: It is a complex decision to determine the highest supported opset version accurately.
    # TODO: Please make this decision more robust and accurate.
    from torch.onnx import _constants
    opset_versions = [v for v in get_onnx_opset_versions() if 14 <= v and v <= _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET]

    for remote_sb3_model_path in remote_sb3_model_paths:

        try:
            logger.info(f'[{model_id}] Downloading sb3 artifact: {remote_sb3_model_path}')
            sb3_model_path = pathlib.Path(hf_hub_download(model_id, remote_sb3_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            logger.error(f'Fail in convert_sb3: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[remote_sb3_model_path] = 'access_deny'
            continue

        status[remote_sb3_model_path] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(sb3_model_path))}.onnx')
        for onnx_opset_version in opset_versions:
            logger.info(f'[{model_id}] [opset {onnx_opset_version}] Starting sb3 export: {remote_sb3_model_path}')
            results_queue = multiprocessing.Queue()
            subprocess = export_process(target=safe_sb3_export, args=(model_id, sb3_model_path, onnx_model_path, onnx_opset_version, results_queue, verbose))
            subprocess.start()
            subprocess.join()

            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {remote_sb3_model_path}')
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(pathlib.Path(onnx_model_path))))
                        instances.append(instance)
                        artifacts.append((remote_sb3_model_path, onnx_opset_version))
                    except Exception as exception:
                        logger.error(f'Fail in convert_sb3: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                        this_status = 'logicx_error'

            log_status = logger.info if this_status == 'success' else logger.warning
            log_status(f'[{model_id}] [opset {onnx_opset_version}] {remote_sb3_model_path}: {this_status}')
            status[remote_sb3_model_path][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_keras_export(keras_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, verbose: bool = False):
    try:
        with export_output(verbose):
            from .miscs import tf2onnx_main_export

            if keras_model_path.is_dir():
                model_type = 'saved_model'
            if keras_model_path.is_file():
                model_type = 'keras'

            tf2onnx_main_export(keras_model_path, onnx_model_path, onnx_opset_version, model_type=model_type)
            this_status = 'success'
    except Exception as exception:
        logger.error(f'Fail to export model: {keras_model_path}; opset: {onnx_opset_version}; {type(exception).__name__}: {exception}', exc_info=verbose)
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_keras(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None, verbose: bool = False) -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[tuple[str, int]]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[tuple[str, int]] = list()
    remote_keras_model_paths = list()
    logger.info(f'[{model_id}] Listing keras model files')
    for remote_keras_model_path in get_huggingface_hub_model_siblings(model_id, suffixes=['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']):
        if remote_keras_model_path.endswith('.pbtxt') or remote_keras_model_path.endswith('.pb'):
            remote_keras_model_paths.append((os.path.dirname(remote_keras_model_path), 'D'))
        else:
            remote_keras_model_paths.append((remote_keras_model_path, 'F'))
    remote_keras_model_paths = list(set(remote_keras_model_paths))

    for remote_keras_model_path, path_type in remote_keras_model_paths:
        artifact = remote_keras_model_path or '.'
        logger.info(f'[{model_id}] Downloading keras artifact: {artifact}')
        try:
            if path_type == 'D':
                allowed_pattern = '*' if remote_keras_model_path == '' else f'{remote_keras_model_path}/*'
                keras_model_path = pathlib.Path(snapshot_download(model_id, allow_patterns=allowed_pattern, cache_dir=ofc_cache_dirpath)).joinpath(remote_keras_model_path)
            if path_type == 'F':
                keras_model_path = pathlib.Path(hf_hub_download(model_id, remote_keras_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            logger.error(f'Fail in convert_keras: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[artifact] = 'access_deny'
            continue

        status[artifact] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(keras_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            logger.info(f'[{model_id}] [opset {onnx_opset_version}] Starting keras export: {artifact}')
            results_queue = multiprocessing.Queue()
            subprocess = export_process(target=safe_keras_export, args=(keras_model_path, onnx_model_path, onnx_opset_version, results_queue, verbose))
            subprocess.start()
            subprocess.join()

            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {artifact}')
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                        artifacts.append((artifact, onnx_opset_version))
                    except Exception as exception:
                        logger.error(f'Fail in convert_keras: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                        this_status = 'logicx_error'
            log_status = logger.info if this_status == 'success' else logger.warning
            log_status(f'[{model_id}] [opset {onnx_opset_version}] {artifact}: {this_status}')
            status[artifact][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_tflite_export(tflite_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, verbose: bool = False):
    try:
        with export_output(verbose):
            from .miscs import tf2onnx_main_export

            tf2onnx_main_export(tflite_model_path, onnx_model_path, onnx_opset_version, model_type='tflite')
            this_status = 'success'
    except Exception as exception:
        logger.error(f'Fail to export model: {tflite_model_path}; opset: {onnx_opset_version}; {type(exception).__name__}: {exception}', exc_info=verbose)
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_tflite(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None, verbose: bool = False) -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[tuple[str, int]]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[tuple[str, int]] = list()
    logger.info(f'[{model_id}] Listing tflite model files')
    remote_tflite_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.tflite'])
    for remote_tflite_model_path in remote_tflite_model_paths:
        logger.info(f'[{model_id}] Downloading tflite artifact: {remote_tflite_model_path}')
        try:
            tflite_model_path = pathlib.Path(hf_hub_download(model_id, remote_tflite_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            logger.error(f'Fail in convert_tflite: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[remote_tflite_model_path] = 'access_deny'
            continue

        status[remote_tflite_model_path] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(tflite_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            logger.info(f'[{model_id}] [opset {onnx_opset_version}] Starting tflite export: {remote_tflite_model_path}')
            results_queue = multiprocessing.Queue()
            subprocess = export_process(target=safe_tflite_export, args=(tflite_model_path, onnx_model_path, onnx_opset_version, results_queue, verbose))
            subprocess.start()
            subprocess.join()
            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {remote_tflite_model_path}')
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                        artifacts.append((remote_tflite_model_path, onnx_opset_version))
                    except Exception as exception:
                        logger.error(f'Fail in convert_tflite: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                        this_status = 'logicx_error'

            log_status = logger.info if this_status == 'success' else logger.warning
            log_status(f'[{model_id}] [opset {onnx_opset_version}] {remote_tflite_model_path}: {this_status}')
            status[remote_tflite_model_path][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_onnx_export(origin_version_onnx_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version, results_queue: multiprocessing.Queue, verbose: bool = False):
    try:
        with export_output(verbose):
            origin_version_onnx_model = load_model(origin_version_onnx_model_path)
            # Convert the ONNX model to the target opset version. This step will not occupy disk space. Thus cvt_cache_dirpath is not used.
            onnx.save_model(onnx.version_converter.convert_version(origin_version_onnx_model, onnx_opset_version), onnx_model_path)
            this_status = 'success'
    except Exception as exception:
        logger.error(f'Fail to export model: {origin_version_onnx_model_path}; opset: {onnx_opset_version}; {type(exception).__name__}: {exception}', exc_info=verbose)
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_onnx(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None, verbose: bool = False) -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[tuple[str, int]]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    artifacts: list[tuple[str, int]] = list()
    logger.info(f'[{model_id}] Listing onnx model files')
    remote_onnx_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.onnx'])
    for remote_onnx_model_path in remote_onnx_model_paths:
        logger.info(f'[{model_id}] Downloading onnx artifact: {remote_onnx_model_path}')
        try:
            onnx_model_path = pathlib.Path(hf_hub_download(model_id, remote_onnx_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            logger.error(f'Fail in convert_onnx: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[remote_onnx_model_path] = 'access_deny'
            continue

        status[remote_onnx_model_path] = dict()

        try:
            logger.info(f'[{model_id}] Loading ONNX artifact: {remote_onnx_model_path}')
            onnx_model = load_model(onnx_model_path)
        except Exception as exception:
            logger.error(f'Fail in convert_onnx: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[remote_onnx_model_path] = 'onnx_load_error'
            continue

        try:
            logger.info(f'[{model_id}] Reading ONNX opset: {remote_onnx_model_path}')
            onnx_model_opset_version = get_onnx_model_opset_version(onnx_model)
        except Exception as exception:
            logger.error(f'Fail in convert_onnx: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
            status[remote_onnx_model_path] = 'onnx_opset_error'
            continue

        for onnx_opset_version in get_onnx_opset_versions():
            if onnx_opset_version == onnx_model_opset_version:
                this_status = 'success'
                try:
                    logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {remote_onnx_model_path}')
                    instance = Instance()
                    instance.setup_logicx(convert(onnx_model))
                    instances.append(instance)
                    artifacts.append((remote_onnx_model_path, onnx_opset_version))
                except Exception as exception:
                    logger.error(f'Fail in convert_onnx: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                    this_status = 'logicx_error'
            else:
                other_version_onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(onnx_model_path))}.onnx')
                logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX opset: {remote_onnx_model_path}')
                results_queue = multiprocessing.Queue()
                subprocess = export_process(target=safe_onnx_export, args=(onnx_model_path, other_version_onnx_model_path, onnx_opset_version, results_queue, verbose))
                subprocess.start()
                subprocess.join()
                if results_queue.empty():
                    this_status = 'system_kill'
                else:
                    this_status = results_queue.get()
                    if this_status == 'success':
                        try:
                            logger.info(f'[{model_id}] [opset {onnx_opset_version}] Converting ONNX to IR: {remote_onnx_model_path}')
                            instance = Instance()
                            instance.setup_logicx(convert(load_model(other_version_onnx_model_path)))
                            instances.append(instance)
                            artifacts.append((remote_onnx_model_path, onnx_opset_version))
                        except Exception as exception:
                            logger.error(f'Fail in convert_onnx: {model_id}; {type(exception).__name__}: {exception}', exc_info=verbose)
                            this_status = 'logicx_error'
            log_status = logger.info if this_status == 'success' else logger.warning
            log_status(f'[{model_id}] [opset {onnx_opset_version}] {remote_onnx_model_path}: {this_status}')
            status[remote_onnx_model_path][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def get_model_infos_and_convert_method(model_infos_filepath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], library_name: str | None = None) -> tuple[list[dict[str, Any]], Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[tuple[str, int]]]]]:
    # This is the convert order.
    supported_frameworks: list[str] = ['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']
    supported_convert_methods: dict[str, Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda'], str | None], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[tuple[str, int]]]]] = dict(
        optimum=convert_optimum,
        onnx=convert_onnx,
        keras=convert_keras,
        tflite=convert_tflite,
        stable_baselines3=convert_sb3,
    )

    def get_model_frameworks(model_frameworks: list[Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']]) -> list[Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']]:
        candidate_frameworks = set(model_frameworks) & set(supported_frameworks)
        model_frameworks = list()
        for supported_framework in supported_frameworks:
            if supported_framework in candidate_frameworks:
                model_frameworks.append(supported_framework)
        return model_frameworks

    model_infos: list[dict[str, Any]] = list()
    for model_info in load_json(model_infos_filepath):
        model_frameworks = get_model_frameworks(infer_supported_frameworks(model_info))
        if library_name is not None and get_model_library_name(model_info) != library_name:
            continue
        if framework in model_frameworks:
            model_infos.append(model_info)

    convert_method = supported_convert_methods[framework]
    return model_infos, convert_method


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], model_size_limit: tuple[int, int], library_name: str | None = None) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    status_name = framework if library_name is None else f'{framework}_{library_name}'
    specific_status_filepath = sts_cache_dirpath.joinpath(f'{status_name}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    if specific_status_filepath.is_file():
        with open(specific_status_filepath, 'r') as specific_status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in specific_status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{status_name}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], model_size_limit: tuple[int, int], convert_status: dict[str, dict[str, Any]], model_id: str, library_name: str | None = None):
    # TODO: The typing notation of the convert_status should be revised in future.
    status_name = framework if library_name is None else f'{framework}_{library_name}'
    convert_status_filepath = sts_cache_dirpath.joinpath(f'{status_name}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{status_name}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


def main(
    model_infos_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
    device: Literal['cpu', 'cuda'] = 'cpu',
    framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'] = 'optimum',
    model_size_limit_l: int | None = None,
    model_size_limit_r: int | None = None,
    token: str | None = None,
    estimate: bool = False,
    library_name: Literal['transformers', 'diffusers', 'timm', 'sentence_transformers'] | None = None,
    verbose: bool = False,
    api_rate_limit: int = 2500,
    resolver_rate_limit: int = 12000,
    machine_count: int = 5,
    rate_limit_utilization: float = 0.8,
):
    """
    Retrieve Metadata of HuggingFace Models and Save Them Into Files.

    :param model_ids_filepath: _description_
    :type model_ids_filepath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    :param cache_dirpath: _description_
    :type cache_dirpath: pathlib.Path
    :param device: _description_, defaults to 'cpu'
    :type device: Literal[&#39;cpu&#39;, &#39;cuda&#39;], optional
    :param framework: _description_, defaults to 'optimum'
    :type framework: Literal[&#39;optimum&#39;, &#39;onnx&#39;, &#39;keras&#39;, &#39;tflite&#39;], optional
    :param verbose: Show backend output and full exception tracebacks for all conversion frameworks.
    :param library_name: Required for optimum; only convert models belonging to this library.
    :param api_rate_limit: Account-wide authenticated API request quota per five minutes.
    :param resolver_rate_limit: Account-wide authenticated Resolver quota per five minutes.
    :param machine_count: Machines sharing the account quotas; use identical settings on each.
    :param rate_limit_utilization: Fraction of each quota to use, in (0, 1].
    :param model_size_limit_l: Inclusive lower size limit in bytes, defaults to None
    :type model_size_limit_l: int | None, optional
    :param model_size_limit_r: Inclusive upper size limit in bytes, defaults to None
    :type model_size_limit_r: int | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    In this project we have a concept called Origin. Origin is a tuple of (hub, owner, name).

    HuggingFace Hub is a place where people can share their models, datasets, and scripts.
    This project hardcodes the hub as 'HuggingFace'.
    The Naming Convention of the Mdoel ID on HuggingFace Hub follows the format: {owner}/{name}.
    Thus the Origin of a Implementation, often called as Model which is a instance of a LogicX a.k.a. Neural Network Architecture (NNA), on HuggingFace Hub is Origin('HuggingFace', owner, name).

    Model Infos are retrieved from the HuggingFace Hub and sorted with lastModified time in descending order, and saved into a JSON file by using command `younger-logics-ir create onnx retrieve huggingface --mode Model_Infos --save-dirpath ${SAVE_DIRPATH}`.

    .. note::
        Instances are saved under Instances/{owner}/{model}/{framework}/opset_{version}/{instance_unique}.
        The model README is saved under Instances/{owner}/{model}/README.md.

    """

    rate_config = RateLimitConfig(api_rate_limit, resolver_rate_limit, machine_count, rate_limit_utilization)

    if framework == 'optimum' and library_name is None:
        raise ValueError('library_name is required when framework is optimum.')
    if library_name is not None:
        if framework != 'optimum':
            raise ValueError('library_name is only available when framework is optimum.')
        if library_name not in {'transformers', 'diffusers', 'timm', 'sentence_transformers'}:
            raise ValueError(f'Unsupported library_name: {library_name}')

    model_size_limit_l = 0 if model_size_limit_l is None else model_size_limit_l
    logger.info(f'   Model Size Left Limit: {get_human_readable_size_representation(model_size_limit_l)}.')

    model_size_limit_r = 1024 ** 5 if model_size_limit_r is None else model_size_limit_r
    logger.info(f'   Model Size Right Limit: {get_human_readable_size_representation(model_size_limit_r)}.')

    model_size_limit = (model_size_limit_l, model_size_limit_r)

    model_infos, convert_method = get_model_infos_and_convert_method(model_infos_filepath, framework, library_name=library_name)
    if estimate:
        logger.info(f'Only Estimate. Models To Be Converted: {len(model_infos)}; Model Infos Filename: {model_infos_filepath.name}.')
        return

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFOfc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFSts')
    create_dir(sts_cache_dirpath)
    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit, library_name=library_name)
    number_of_converted_models = len(convert_status)
    logger.info(f'-> Previous Converted Models: {number_of_converted_models}')

    configure_hub_http(config=rate_config)
    budgets = rate_config.budgets()
    logger.info(f'-> Hub request pacing: {machine_count} machines, utilization {rate_limit_utilization:.0%}; '
                f'per machine API {budgets["api"]:g}/5min, Resolver {budgets["resolvers"]:g}/5min.')

    if token is not None:
        logger.info(f'-> HuggingFace Token Provided. Now Logging In ...')
        login(token)
    else:
        logger.info(f'-> HuggingFace Token Not Provided. Now Accessing Without Token ...')

    logger.info(f'-> Instances Creating ...')
    with tqdm.tqdm(total=len(model_infos), desc='Create Instances') as progress_bar:
        for convert_index, model_info in enumerate(model_infos, start=1):
            model_id = model_info['id']
            progress_bar.set_description(f'Checking - {model_id}')
            logger.info(f'[{convert_index}/{len(model_infos)}] Checking model storage: {model_id}')
            possible_model_size = get_huggingface_hub_model_storage(model_id, token=token)
            if model_size_limit_l <= possible_model_size and possible_model_size <= model_size_limit_r:
                model_library_name = get_model_library_name(model_info) if framework == 'optimum' else None
                if last_handled_model_id is not None:
                    if model_id == last_handled_model_id:
                        last_handled_model_id = None
                    progress_bar.set_description(f'Converted, Skip - {model_id}')
                    progress_bar.update(1)
                    continue

                progress_bar.set_description(f'Converting - {model_id}')
                logger.info(f'[{convert_index}/{len(model_infos)}] Starting conversion: {model_id}; framework: {framework}; library: {model_library_name}; device: {device}')
                status, instances, artifacts = convert_method(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device, library_name=model_library_name, verbose=verbose)
                logger.info(f'[{model_id}] Conversion returned {len(instances)} instances; saving results')

                model_owner, model_name = model_id.split('/')
                model_dirpath = instances_dirpath.joinpath(model_owner, model_name)
                saved_instances = 0
                for index, (instance, (artifact, onnx_opset_version)) in enumerate(zip(instances, artifacts), start=1):
                    instance.insert_label(
                        Implementation(
                            origin=Origin(YLIROriginHub.HUGGINGFACE, model_owner, model_name, artifact),
                            like=model_info['likes'],
                            download=model_info['downloadsAllTime'],
                        )
                    )
                    instance_unique = instance.unique
                    instance_dirpath = model_dirpath.joinpath(framework, f'opset_{onnx_opset_version}', instance_unique)
                    try:
                        instance.save(instance_dirpath)
                        save_json(
                            dict(
                                model_id=model_id,
                                framework=framework, library_name=model_library_name,
                                artifact=artifact,
                                opset_version=onnx_opset_version,
                                opset_import=instance.logicx.dag.graph.get('opset_import', {})
                            ),
                            instance_dirpath.joinpath('conversion.json'), indent=2
                        )
                        saved_instances += 1
                    except FileExistsError as exception:
                        logger.warning(f'-> Skip! Instance Already Exists: {instance_unique}')

                if saved_instances:
                    try:
                        readme = get_huggingface_hub_model_readme(model_id, token=token)
                    except Exception as exception:
                        readme = ''

                    if readme == '':
                        pass
                    else:
                        model_dirpath.joinpath('README.md').write_text(readme, encoding='utf-8')
            else:
                status = dict(special_status='size_limit')

            set_convert_status_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit, status, model_id, library_name=library_name)
            clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)

            progress_bar.set_description(f'Handled - {model_id}')
            logger.info(f'[{convert_index}/{len(model_infos)}] Finished handling model: {model_id}')
            progress_bar.update(1)

    logger.info(f'-> Instances Created.')
