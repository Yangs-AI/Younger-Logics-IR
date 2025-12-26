#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-12-25 02:10:59
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-25 21:51:53
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


class Schema(object):
    def __init__(self,
        **kwargs
    ) -> None:
        if kwargs.get('id', None) is not None:
            self.id = kwargs['id']

    def dict(self):
        items = dict()
        for key, value in self.__dict__.items():
            if key == 'id':
                continue
            if not key.startswith('_'):
                items[key] = value
        return items
