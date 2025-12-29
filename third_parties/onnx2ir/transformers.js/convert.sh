#!/bin/bash

# Created time: 2024-08-27 18:03:44
# Author: Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-27 15:35:31
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.


if [ "$#" -ne 3 ]; then
    echo "Usage:"
    echo "    ./convert.sh {model_ids_filepath} {cache_dirpath} {save_dirpath}"
    exit 1
fi

MODEL_IDS_FILEPATH=${1}
CACHE_DIRPATH=${2}
SAVE_DIRPATH=${3}

model_ids=($(jq -r '.[]' "${MODEL_IDS_FILEPATH}"))

for ((i=0; i<${#model_ids[@]}; i++)); do
    model_id=${model_ids[i]}
    python -m convert.main --quantize --model_id "${model_id}" --trust_remote_code --skip_validation --onnx_output_dir "${CACHE_DIRPATH}" --instance_output_dir "${SAVE_DIRPATH}" --clean_onnx_dir
done
