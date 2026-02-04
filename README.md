# Younger-Logics-IR
Younger - Logics - Intermediate Representation

Younger-Logics-IR is a submodule of the Younger project designed to construct a unified Intermediate Representation (IR) for abstracting and describing the logic structure of deep learning models. It supports extracting model information from various frameworks (e.g., PyTorch and ONNX) and abstracts the operators and data flows into a unified graph structure. Younger-Logics-IR enables framework-agnostic model analysis, conversion, and optimization, serving as a foundational tool for deep learning workflows.

## Optional Dependencies (Extras)
This package defines several optional dependency groups for different scripts/tools.

Install an extra with:

```
pip install "younger-logics-ir[EXTRA_NAME]"
```

Available extras:

- `scripts-hubs-hf`: Crawl models and metadata from Hugging Face Hub and build Younger IR-Dataset (FromONNX). Each sample is organized as LogicX or Instance (DAG + related info).
- `scripts-hubs-ox`: Crawl models and metadata from the ONNX Model Zoo. This is no longer maintained since most models moved to Hugging Face Hub.
- `scripts-hubs-tr`: Crawl models and metadata from Torch Hub. This hub contains very few models compared to Hugging Face Hub.
- `scripts-hubs`: Full hub stack (combined set).
- `scripts-bootstrap`: Filter, normalize, post-process, and some other feature engineering on collected DAG datasets.
- `tools-vs`: Visualize DAGs (Graphviz).
- `tools`: Full toolset (combined set).
- `develop`: Developer tools (docs, pytest, release tooling).


### For the Use of Assorts
#### Requirements

`Graphviz` must be installed:

* Mac
```
brew install graphviz
```

* Debian/Ubuntu
```
sudo apt install graphviz
```

* Conda
```
conda install conda-forge::python-graphviz
```

##### Usage

```
younger logics ir create onnx retrieve huggingface \
    --mode Metric_Infos \
    --save-dirpath /path/to/working_directory/ \
    --token <HF_API_TOKEN> \
    --number-per-file 100000 \
    --logging-filepath /path/to/working_directory/.younger.log
```
