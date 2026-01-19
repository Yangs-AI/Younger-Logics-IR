#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-19 22:12:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-19 10:04:04
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import copy
import pathlib
import networkx

from typing import Literal, Generator

from younger.commons.io import load_pickle, save_pickle, loads_pickle, saves_pickle, loads_json, saves_json, get_object_with_sorted_dict
from younger.commons.hash import hash_string

from younger_logics_ir.commons.json import YLIRJSONEncoder, YLIRJSONDecoder


class LogicX(object):

    __DAGHead__ = 'YLIR-DAG-'

    def __init__(self, src: Literal['onnx', 'code'] | None = None, dag: networkx.DiGraph | None = None):
        assert src in {'onnx', 'code'} or src is None, f'Argument \"src\" must be in {{"onnx", "code"}} instead \"{type(src)}\"!'
        assert isinstance(dag, networkx.DiGraph) or dag is None, f'Argument \"dag\" must be `networkx.DiGraph` instead \"{type(dag)}\"!'
        self._src = src
        self._dag = dag

        # If the LogicX is not standardized, the self._relationship is None.
        # If the LogicX is standardized:
        #     - If the LogicX is a sub-LogicX, the self._relationship is its parent's hash.
        #     - If the LogicX has no parent, the self._relationship is its own hash.
        self._relationship: str | None = None

    @property
    def valid(self) -> bool:
        return self._dag is not None and self._src is not None

    @property
    def src_valid(self) -> bool:
        return self._src is not None

    @property
    def dag_valid(self) -> bool:
        return self._dag is not None

    @property
    def src(self) -> Literal['onnx', 'code'] | None:
        return self._src

    @property
    def dag(self) -> networkx.DiGraph | None:
        return self._dag

    def setup_src(self, src: Literal['onnx', 'code']) -> None:
        assert src in {'onnx', 'code'}, f'Argument \"src\" must be in {{"onnx", "code"}} instead \"{type(src)}\"!'
        self._src = src

    def setup_dag(self, dag: networkx.DiGraph) -> None:
        assert isinstance(dag, networkx.DiGraph), f'Argument \"dag\" must be `networkx.DiGraph` instead \"{type(src)}\"!'
        self._dag = dag

    def load(self, logicx_filepath: pathlib.Path) -> None:
        assert logicx_filepath.is_file(), f'There is no \"LogicX\" can be loaded from the specified path \"{logicx_filepath.absolute()}\".'
        (ldag, lsrc, lrelationship) = load_pickle(logicx_filepath)
        self._relationship = loads_pickle(lrelationship)
        self._src = loads_pickle(lsrc)
        self._dag = self.__class__.loads_dag(ldag)
        return

    def save(self, logicx_filepath: pathlib.Path) -> None:
        assert not logicx_filepath.is_file(), f'\"LogicX\" can not be saved into the specified path \"{logicx_filepath.absolute()}\".'
        sdag = self.__class__.saves_dag(self._dag)
        ssrc = saves_pickle(self._src)
        srelationship = saves_pickle(self._relationship)
        save_pickle((sdag, ssrc, srelationship), logicx_filepath)
        return

    def node_features(self, node_index: str) -> dict[str, str | dict]:
        """
        Get node features by node index.

        :param node_index: The index of the node.
        :type node_index: str

        :return: Node features dictionary containing node type and attributes.
        :rtype: dict[str, str | dict | int]

        Format - {node_type: str, node_attr: dict}
        """
        assert self.valid, f'\"LogicX\" is invalid!'
        return self._dag.nodes[node_index]

    def node_tuid_feature(self, node_index: str) -> str:
        assert self.valid, f'\"LogicX\" is invalid!'
        return self._dag.nodes[node_index]['node_tuid']

    def node_type_feature(self, node_index: str) -> str:
        assert self.valid, f'\"LogicX\" is invalid!'
        return self._dag.nodes[node_index]['node_type']

    def node_name_feature(self, node_index: str) -> str | None:
        assert self.valid, f'\"LogicX\" is invalid!'
        return self._dag.nodes[node_index].get('node_name', None)

    def node_attr_feature(self, node_index: str) -> dict | None:
        assert self.valid, f'\"LogicX\" is invalid!'
        return self._dag.nodes[node_index].get('node_attr', None)

    def node_indices(self, node_type: Literal['input', 'output', 'operator', 'outer']) -> Generator[str, None, None]:
        assert self.valid, f'\"LogicX\" is invalid!'
        for node_index in self._dag.nodes:
            if self._dag.nodes[node_index]['node_type'] == node_type:
                yield node_index
            else:
                continue

    def edge_indices(self) -> Generator[tuple[str, str], None, None]:
        assert self.valid, f'\"LogicX\" is invalid!'
        for edge_u_index, edge_v_index in self._dag.edges:
            yield (edge_u_index, edge_v_index)

    @property
    def input_indices(self) -> Generator[str, None, None]:
        return self.node_indices('input')

    @property
    def output_indices(self) -> Generator[str, None, None]:
        return self.node_indices('output')

    @property
    def operator_indices(self) -> Generator[str, None, None]:
        return self.node_indices('operator')

    @property
    def outer_indices(self) -> Generator[str, None, None]:
        return self.node_indices('outer')

    @property
    def standard(self) -> bool:
        return not (self._relationship is None)

    @property
    def relationship(self) -> str:
        assert self.standard, f'\"LogicX\" is not standardized!'
        return self._relationship

    @classmethod
    def loads_dag(cls, txt: str) -> networkx.DiGraph:
        dic = loads_json(txt, cls=YLIRJSONDecoder)
        dag = cls.loadd_dag(dic)
        return dag

    @classmethod
    def saves_dag(cls, dag: networkx.DiGraph) -> str:
        dic = cls.saved_dag(dag)
        txt = saves_json(dic, cls=YLIRJSONEncoder)
        return txt

    @classmethod
    def loadd_dag(cls, dic: dict) -> networkx.DiGraph:
        def str2dag(obj: object) -> object:
            if isinstance(obj, dict):
                return {key: str2dag(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [str2dag(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith(cls.__DAGHead__):
                return cls.loads_dag(obj[len(cls.__DAGHead__):])
            else:
                return obj

        dag = networkx.DiGraph()

        dag.graph = dic.get("graph", dict())
        for node_data in dic['nodes']:
            node_index = node_data['node_index']
            node_features = str2dag(node_data['node_features'])
            dag.add_node(node_index, **node_features)
        for edge_data in dic['edges']:
            tail_index = edge_data['tail_index']
            head_index = edge_data['head_index']
            edge_features = edge_data['edge_features']
            dag.add_edge(tail_index, head_index, **edge_features)

        return dag

    @classmethod
    def saved_dag(cls, dag: networkx.DiGraph) -> dict:
        def dag2str(obj: object) -> object:
            if isinstance(obj, dict):
                return {key: dag2str(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [dag2str(item) for item in obj]
            elif isinstance(obj, networkx.DiGraph):
                return f'{cls.__DAGHead__}{cls.saves_dag(obj)}'
            else:
                return obj

        dic = dict(
            graph=dag.graph,
            nodes=[
                dict(
                    node_features=get_object_with_sorted_dict(dag2str(node_features)),
                    node_index=node_index,
                ) for node_index, node_features in sorted(dag.nodes(data=True), key=lambda x: x[0])
            ],
            edges=[
                dict(
                    edge_features=get_object_with_sorted_dict(edge_features),
                    tail_index=tail_index,
                    head_index=head_index,
                ) for tail_index, head_index, edge_features in sorted(dag.edges(data=True), key=lambda x: (x[0], x[1]))
            ]
        )

        return dic

    @classmethod
    def simplify(cls, logicx: 'LogicX') -> tuple['LogicX', list['LogicX']]:
        logicx_sons: list['LogicX'] = list()
        logicx_descendants: list['LogicX'] = list()
        for operator_index in logicx.operator_indices:
            operator_attributes: dict[str, dict] = logicx.node_attr_feature(operator_index)['attributes']
            for oa_name, oa_attr in operator_attributes.items():
                if isinstance(oa_attr['value'], networkx.DiGraph):
                    logicx_son, logicx_son_descendants = cls.simplify(LogicX(logicx.src, oa_attr['value']))
                    logicx_sons.append(logicx_son)
                    logicx_descendants.extend(logicx_son_descendants)
                    operator_attributes[oa_name]['value'] = cls.luid(logicx_son)

                if isinstance(oa_attr['value'], list) and len(oa_attr['value'])!=0 and isinstance(oa_attr['value'][0], networkx.DiGraph):
                    assert all(isinstance(possible_subdag, networkx.DiGraph) for possible_subdag in oa_attr['value']), f'All subdags should be `networkx.DiGraph`!'
                    new_oa_attr_value: list[str] = list() # All change to hash
                    for subdag in oa_attr['value']:
                        logicx_son, logicx_son_descendants = cls.simplify(LogicX(logicx.src, subdag))
                        logicx_sons.append(logicx_son)
                        logicx_descendants.extend(logicx_son_descendants)
                        new_oa_attr_value.append(cls.luid(logicx_son))
                    operator_attributes[oa_name]['value'] = new_oa_attr_value

        logicx_hash = cls.luid(logicx)
        for logicx_son in logicx_sons:
            logicx_son._relationship = logicx_hash

        logicx._relationship = logicx_hash
        return logicx, logicx_sons + logicx_descendants

    @classmethod
    def standardize(cls, logicx: 'LogicX') -> 'LogicX':
        """
        Standardize LogicX.

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: LogicX
        """
        assert logicx.standard, f'\"LogicX\" is not simplified!'
        logicx = cls.copy(logicx)
        src = logicx.src
        dag = networkx.DiGraph()
        for node_index in logicx.dag.nodes():
            node_tuid = logicx.node_tuid_feature(node_index)
            node_type = logicx.node_type_feature(node_index)
            if node_type == 'operator':
                node_attr = logicx.node_attr_feature(node_index)['attributes']
                for node_attr_name, node_attr_value in node_attr.items():
                    node_attr_value.pop('doc_string')
            else:
                node_attr = dict()
            node_feat = dict(
                node_tuid = node_tuid,
                node_type = node_type,
                node_attr = node_attr,
            )
            node_uuid = hash_string(saves_json(get_object_with_sorted_dict(node_feat), cls=YLIRJSONEncoder))
            dag.add_node(node_index, node_uuid=node_uuid, node_tuid=node_tuid, node_type=node_type, node_attr=node_attr)

        for edge_u_index, edge_v_index in logicx.dag.edges():
            dag.add_edge(edge_u_index, edge_v_index)

        logicx_standard = LogicX(src, dag)
        return logicx_standard

    @classmethod
    def skeletonize(cls, logicx: 'LogicX') -> 'LogicX':
        """
        Skeletonize LogicX.

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: LogicX
        """
        assert logicx.standard, f'\"LogicX\" is not simplified!'
        logicx = cls.copy(logicx)
        src = logicx.src
        dag = networkx.DiGraph()
        for node_index in logicx.dag.nodes():
            node_tuid = logicx.node_tuid_feature(node_index)
            node_type = logicx.node_type_feature(node_index)
            node_feat = dict(
                node_tuid = node_tuid,
                node_type = node_type,
            )
            node_uuid = hash_string(saves_json(get_object_with_sorted_dict(node_feat), cls=YLIRJSONEncoder))
            dag.add_node(node_index, node_uuid=node_uuid, node_tuid=node_tuid, node_type=node_type)

        for edge_u_index, edge_v_index in logicx.dag.edges():
            dag.add_edge(edge_u_index, edge_v_index)

        logicx_skeleton = LogicX(src, dag)
        return logicx_skeleton

    @classmethod
    def hash(cls, logicx: 'LogicX') -> str:
        """
        Topology Hash (Hash)

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: str
        """
        return networkx.weisfeiler_lehman_graph_hash(logicx.dag, edge_attr=None, node_attr='node_uuid', iterations=3, digest_size=16)

    @classmethod
    def luid(cls, logicx: 'LogicX') -> str:
        """
        LogicX Unique ID (LUID)

        :param logicx: _description_
        :type logicx: LogicX

        :return: _description_
        :rtype: str
        """

        luid = hash_string(cls.saves_dag(logicx.dag))
        return luid

    @classmethod
    def copy(cls, logicx: 'LogicX') -> 'LogicX':
        """
        Copy LogicX.
        .. todo::
            This project want deepcopy and we want to check the performance of deepcopy.

        :param logicx: _description_
        :type logicx: LogicX
        :return: _description_
        :rtype: LogicX
        """
        logicx_copy = LogicX(logicx.src, copy.deepcopy(logicx.dag))
        return logicx_copy
