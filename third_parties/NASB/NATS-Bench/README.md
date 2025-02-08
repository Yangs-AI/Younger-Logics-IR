# NAS-Bench-101 to YLIR
This project is based on the repositorys "NATS-Bench" (https://github.com/D-X-Y/NATS-Bench) 
and AutoDL-Projects (https://github.com/D-X-Y/AutoDL-Projects), 
with modifications for converting NATS-Bench models to Younger.

NATS-Bench official repository: [https://github.com/google-research/nasbench
](https://github.com/D-X-Y/NATS-Bench)

To run this project, you need to have the following dependencies installed:

- **Python** version >= 3.11
- **Pytorch** version == 2.1.0

First, install `Younger-Logics-IR`.
```bash
pip install younger-logics-ir[scripts]
```

Next, install `nats_bench`.
```bash
pip install nats_bench
```

After installing the required packages, you can run the conversion code `convert.py`.

The `model-infos-dirpath` must be set to either `NATS-tss-v1_0-3ffb9-simple` or `NATS-sss-v1_0-50262-simple` (topology or size). Download the (unpacked benchmark file) archive from [https://github.com/google-research/nasbench
](https://github.com/D-X-Y/NATS-Bench) and then uncompress it by using `tar xvf [file_name]` to get these two directories mentioned above.

**Hint**: The topology search space in NATS-Bench is the same as NAS-Bench-201!

Run command as below (taking NATS-topology as an example):
```bash
python convert.py --model-infos-dirpath /path/to/NATS-tss-v1_0-3ffb9-simple --save-dirpath /path/to/save --cache-dirpath /path/to/cache --search-space-type tss --start-index 0 --end-index 10 --opset 15 --worker-number 8
```
