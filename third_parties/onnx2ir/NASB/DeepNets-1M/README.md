# DeepNets-1M to YLIR
This project is based on the repository "ppuda" (https://github.com/facebookresearch/ppuda), with modifications for converting DeepNets-1M models to YLIR instances.

**Hint**: The split "train" in code corresponds to DeepNets-1M dataset.

ppuda official repository: https://github.com/facebookresearch/ppuda

To run this project, you need to have the following dependencies installed:

- **Python** version >= 3.11
- **PyTorch** version == 2.1.0
- **numpy** version == 1.26.4

First, install `Younger-Logics-IR`.
```bash
pip install younger-logics-ir[scripts]
```

Next, install `ppuda`:
```bash
git clone https://github.com/facebookresearch/ppuda.git
cd path/to/ppuda
pip install -e .
```

To accelerate and simplify the conversion process, run `generate_model_infos.py` to generate `DeepNets-1M_Model_Infos_${i}.pth`. The `data-dir` should contain dataset files download from [ppuda/data/download.sh](https://github.com/facebookresearch/ppuda/blob/main/data/download.sh) in `ppuda`. The train split corresponds to DeepNets-1M dataset.

Run command as below:
```bash
python generate_model_infos.py --save-dirpath path/to/save/model_infos --data-dirpath path/to/ppuda_data --interval 50000 --split train
```

After generating the `DeepNets-1M_Model_Infos_${i}.pth`, you can run the conversion code `convert.py`.

Run command as below:
```bash
python convert.py --model-infos-filepath /path/to/DeepNets-1M_Model_Infos_${i}.pth --save-dirpath /path/to/save --cache-dirpath /path/to/cache --start-index 0 --end-index 10 --opset 18 --worker-number 8
```
