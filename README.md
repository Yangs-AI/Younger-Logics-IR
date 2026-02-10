# Younger-Logics-IR
Younger - Logics - Intermediate Representation

Younger-Logics-IR is a submodule of the Younger project designed to construct a unified Intermediate Representation (IR) for abstracting and describing the logic structure of deep learning models. It supports extracting model information from various frameworks (e.g., PyTorch and ONNX) and abstracts the operators and data flows into a unified graph structure. Younger-Logics-IR enables framework-agnostic model analysis, conversion, and optimization, serving as a foundational tool for deep learning workflows.

## Contributing
This repository is a submodule of the parent project. For the canonical workflow and submodule pointer rules, see [CONTRIBUTING.md](https://github.com/Yangs-AI/Younger/CONTRIBUTING.md).

If you are working inside a parent clone:
- Commit changes in this submodule repo and open a PR to the submodule upstream.
- Do not update the parent repo submodule pointer unless a maintainer asks; mention any required pointer bump in your PR description.

If you are working standalone:
- Fork this repo, add `upstream`, create a feature branch, then open a PR back to upstream.

### Git workflow (sync with upstream)
```bash
# 1) Sync with upstream
git remote -v               # confirm origin/upstream are set correctly
git fetch --prune upstream   # sync upstream refs and drop deleted branches
git checkout dev             # ensure you are on the main dev branch
git pull --rebase upstream dev  # replay local dev on top of upstream/dev

# If you keep local commits on dev, rebase explicitly:
git rebase upstream/dev      # same as above, but explicit and easier to debug
# If conflicts happen:
git status                   # see which files are conflicted
# resolve files, then
git add <conflicted-file>    # mark conflicts as resolved
git rebase --continue        # continue replaying commits
# to abort:
git rebase --abort           # roll back to pre-rebase state

# 2) Create a feature branch
git checkout -b feat/my-change

# 3) Make changes and commit
git add .
git commit -m "feat: my change"

# 4) Push to your fork
git push origin feat/my-change
```

### Fixing edits made on a detached HEAD
If you forgot to switch branches and edited files on a `Previous HEAD position`, stash the changes, switch to the target branch, then restore them:

```bash
# 1) Stash the current changes
git stash push -m "temp: bench local edits"

# 2) Switch to dev
git checkout dev

# 3) Restore the changes
git stash pop
```

### Local development
From this repo root:

```bash
# 1) Install with dev tools
pip install -e .[develop]

# 2) Run tests
pytest tests
```

### Common commands
```bash
younger-logics-ir --help
younger logics ir create onnx retrieve huggingface \
    --mode Metric_Infos \
    --save-dirpath /path/to/working_directory/ \
    --token <HF_API_TOKEN> \
    --number-per-file 100000 \
    --logging-filepath /path/to/working_directory/.younger.log
```

### More docs
- [Homepage](https://younger.yangs.ai/logics/ir)

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
