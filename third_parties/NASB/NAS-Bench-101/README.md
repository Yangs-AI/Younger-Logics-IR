# NAS-Bench-101 to YLIR
This project is based on the "nasbench_keras" repository by lienching (https://github.com/lienching/nasbench_keras), with modifications for converting NAS-Bench-101 models to YLIR instances.

NAS-Bench-101 official repository: https://github.com/google-research/nasbench

To run this project, you need to have the following dependencies installed:

- **Python** version >= 3.11
- **TensorFlow** version == 2.15

First, install Younger-Logics-IR.
```bash
pip install younger-logics-ir[scripts]
```

Next, install `nasbench_keras`.
```bash
git clone https://github.com/Yangs-AI/Younger-Logics-IR.git
cd Younger-Logics-IR/third_parties/NASB/NAS-Bench-101
pip install -e .
```

After installing the required packages, you can run the conversion code `convert.py`.

The `model-infos-path` must be set to `NAS-Bench-101_Model_Infos.json`. The `NAS-Bench-101_Model_Infos.json` can be generated using the official NASBench101 code.

Run command as below:
```bash
TF_CPP_MIN_LOG_LEVEL=3 python convert.py --model-infos-filepath /path/to/NAS-Bench-101_Model_Infos.json --save-dirpath /path/to/save --cache-dirpath /path/to/cache --start-index 0 --end-index 10 --opset 15
```

*NOTE*: The conversion time can be reduced if logging messages are suppressed by setting `TF_CPP_MIN_LOG_LEVEL=3`
