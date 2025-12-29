# NATS-Bench-101 to YLIR
This project is based on the repositories "NATS-Bench" (https://github.com/D-X-Y/NATS-Bench) and AutoDL-Projects (https://github.com/D-X-Y/AutoDL-Projects), with modifications for converting NATS-Bench models to YLIR instances.

**Hint**: The `topology` search space in NATS-Bench is the same as NAS-Bench-201!

NATS-Bench official repository: https://github.com/D-X-Y/NATS-Bench

To run this project, you need to have the following dependencies installed:

- **Python** version >= 3.11

First, install `Younger-Logics-IR`.
```bash
pip install younger-logics-ir[scripts]
```

Next, install `nats_bench`.
```bash
pip install nats_bench
```

Last, install `AutoDL-Projects`:
```bash
git clone https://github.com/D-X-Y/AutoDL-Projects.git
cd AutoDL-Projects
pip install -e .
```

After installing the required packages, you can run the conversion code `convert.py`.

The `model-infos-dirpath` must be set to either [`NATS-tss-v1_0-3ffb9-simple`](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) or [`NATS-sss-v1_0-50262-simple`](https://drive.google.com/file/d/1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA/view?usp=sharing) (`topology` or `size`). (You can download the '(unpacked benchmark file) archive' from [Official GitHub Repo](https://github.com/D-X-Y/NATS-Bench) or [Ours BackUp](#) and then uncompress the archive by using `tar xvf [file_name]` to get these two directories mentioned above.)

Run command as below (taking `topology` as an example):
```bash
python convert.py --model-infos-dirpath /path/to/NATS-tss-v1_0-3ffb9-simple --save-dirpath /path/to/save --cache-dirpath /path/to/cache --search-space-type tss --start-index 0 --end-index 10 --opset 18 --worker-number 8
```
