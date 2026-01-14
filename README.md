# Younger-Logics-IR
Younger - Logics - Intermediate Representation

Younger-Logics-IR is a submodule of the Younger project designed to construct a unified Intermediate Representation (IR) for abstracting and describing the logic structure of deep learning models. It supports extracting model information from various frameworks (e.g., PyTorch and ONNX) and abstracts the operators and data flows into a unified graph structure. Younger-Logics-IR enables framework-agnostic model analysis, conversion, and optimization, serving as a foundational tool for deep learning workflows.


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
