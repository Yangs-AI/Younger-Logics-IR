#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:19
# Author: Jason Young (杨郑鑫) and Yikang Yang (杨怡康).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-31 11:16:36
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import re
import requests

from typing import Any

from younger.commons.string import extract_possible_tables_from_readme_string


def get_onnx_hub_model_infos() -> list[dict[str, Any]]:
    response = requests.get("https://github.com/onnx/models/raw/main/README.md", timeout=30)
    if response.status_code == 200:
        readme = response.text
    else:
        raise Exception("Failed to get README.md from onnx/models")

    official_task = [
        "Image Classification",
        "Domain-based Image Classification",
        "Object Detection & Image Segmentation",
        "Body, Face & Gesture Analysis",
        "Image Manipulation",
        "Machine Comprehension",
        "Machine Translation",
        "Language Modelling",
        "Visual Question Answering & Dialog",
        "Speech & Audio Processing",
        "Other interesting models"
    ]

    task2table = {}

    lines = readme.split("\n")
    flag = False
    for line in lines:
        if line.startswith("#"):
            task = line.replace("#", "")
            pattern = r'<a[^>]*>.*?</a>|<a[^>]*/>'
            task = re.sub(pattern, "", task)
            task = task.strip()
            if task in official_task:
                flag = True
                task2table[task] = []
            else:
                flag = False
        else:
            if flag:
                task2table[task].append(line)

    for task in task2table:
        task2table[task] = extract_possible_tables_from_readme_string("\n".join(task2table[task]))

    dir2task = {}
    for task in task2table:
        for table in task2table[task]:  
            for i in range(len(table["rows"])):
                model_class = table["rows"][i][0]
                pattern = r'\[(.*)\]\((validated/.*)\)'
                match = re.search(pattern, model_class)
                if match:
                    dir2task[match.group(2)] = task

    tasks = ["Computer_Vision", "Generative_AI", "Graph_Machine_Learning", "Natural_Language_Processing"]
    authors = ["timm", "torch_hub", "torchvision", "transformers", "graph_convolutions"]

    response = requests.get("https://api.github.com/repos/onnx/models/git/trees/main?recursive=1", timeout=30)

    if response.status_code == 200:
        tree_data = response.json()["tree"]
    else:
        raise Exception("Failed to get tree data from onnx/models")

    model_infos: list[dict[str, Any]] = list()
    for item in tree_data:
        if item["path"].endswith(".onnx"):
            task = item["path"].split("/")[0]
            if task in tasks:
                second_path = item["path"].split("/")[1].strip()
                if second_path == "skip":
                    second_path = item["path"].split("/")[2].strip()
                author = None
                for i in range(len(authors)):
                    if second_path.endswith(authors[i]):
                        author = authors[i]
                second_path = second_path.replace(f"_{author}", "")
                opset = second_path.split("_")[-1]
                model_id = author + "/" + item["path"].split("/")[-1].strip(".onnx")
                model_infos.append(
                    dict(
                        id=model_id, 
                        task=task,
                        opset=int(opset.replace("Opset", "")),
                        url="https://github.com/onnx/models/raw/main/" + item["path"]
                    )
                )
            elif task == "validated":
                for dir in dir2task:
                    dir1 = dir if dir[-1] == "/" else dir + "/"
                    if dir1 in item["path"]:
                        model_id = "official/" + item["path"].split("/")[-1].strip(".onnx")
                        model_infos.append(
                            dict(
                                id=model_id,
                                task=dir2task[dir],
                                opset=None,
                                url="https://github.com/onnx/models/raw/main/" + item["path"]
                            )
                        )

    return model_infos


def get_onnx_hub_model_ids() -> list[str]:
    model_infos = get_onnx_hub_model_infos()
    model_ids = [model_info['id'] for model_info in model_infos]
    return model_ids
