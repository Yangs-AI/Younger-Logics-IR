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

- `scripts-hubs-hf`: Hugging Face metadata retrieval and existing ONNX models; no model export backends.
- `scripts-hubs-hf-transformers`: Transformers export through Optimum.
- `scripts-hubs-hf-diffusers`: Diffusers export through Optimum (also requires Transformers through Optimum).
- `scripts-hubs-hf-timm`: Timm export through Optimum.
- `scripts-hubs-hf-sentence-transformers`: Sentence Transformers export through Optimum.
- `scripts-hubs-hf-tensorflow`: Existing Keras/SavedModel/TFLite conversion with TensorFlow 2.15.0; use Python 3.11.
- `scripts-hubs-hf-sb3`: Stable-Baselines3 conversion.
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


## Releases

This repository publishes its Python package independently to GitHub and PyPI.
The release version is defined in `pyproject.toml`; a matching `v<version>` tag
runs `.github/workflows/release.yml`. Pull requests and manual runs validate the
build without publishing. Internal dependency bounds declare which Younger and
component versions are compatible.

Before the first release, configure the PyPI Trusted Publisher for this repository,
workflow `release.yml`, and GitHub environment `pypi`. Publish required Younger/IR
versions before dependent packages. See the
[routine release checklist](https://github.com/Yangs-AI/Younger/blob/main/RELEASE.md#routine-release)
for setup, dependency order, and combination validation. Conda distribution is
deferred.

For each IR release, update this repository's `pyproject.toml` version and any
required dependency bounds on `dev`, run the relevant tests, and push the changes.
The required Younger version must already be installable from PyPI before IR's
standalone installation check can pass. Open a `dev` to `main` PR and confirm
**Build and release** jobs `build` and `verify` succeeded, then merge it.

From this repository's root (`younger/logics/ir` in a parent checkout), with a
clean working tree and the intended official `origin`, inspect the release:

```bash
git fetch origin --tags
git switch main
git pull --ff-only origin main
git log -1 --oneline
release_version=$(python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')
git tag --list "v${release_version}"
```

Confirm the version and commit. If the tag is already listed, stop and inspect it.
Otherwise, publish this IR version by creating and pushing its tag:

```bash
git tag -a "v${release_version}" -m "Release ${release_version}"
git push origin "v${release_version}"
```

In GitHub **Actions**, confirm the tag run's `build`, `verify`, `publish`, and
`github-release` jobs succeeded. Verify the exact version from PyPI in a clean
environment with `pip check` and `younger-logics-ir --help`. Then update the parent
repository's IR pointer and `releases/current.txt` for the new combination.
Compatible IR fixes do not require matching version bumps in Younger or DL.

## Hugging Face conversion in separate environments

Install the matching `scripts-hubs-hf-*` extra in each environment in one command
so pip resolves the backend dependencies together. Install CPU PyTorch separately
when using Torch Hub.

The Younger kernel and Younger-Logics-IR require Python 3.11 or newer.
Other Younger components retain their own Python requirements. Use Python 3.11
for the existing TensorFlow 2.15.0 conversion path.

Create and activate a separate environment for each model library. For example,
from the parent Younger checkout, prepare a Diffusers environment:

```bash
python3.11 -m venv .venv-hf-diffusers
source .venv-hf-diffusers/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -e 'younger/logics/ir[scripts-hubs-hf-diffusers]'
python -m pip check
mkdir -p output cache-diffusers
younger-logics-ir create onnx convert huggingface \
    --model-infos-filepath ./models.json \
    --save-dirpath ./output \
    --cache-dirpath ./cache-diffusers \
    --framework optimum \
    --library-name diffusers \
    --device cpu
```

The CPU command above uses the PyTorch CPU index for supported platforms; see
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Select compatible CPU torch/torchvision versions for your environment and retain
those versions in a constraints file when installing additional dependencies.
Extras declare required libraries, not a tested lockfile: an isolated Diffusers
environment still needs compatible Optimum, Transformers, Diffusers and Hub versions.

Use the matching extra and `--library-name transformers`, `timm`, or
`sentence_transformers` in the other Optimum environments. The option filters
models using the existing tag priority: sentence-transformers, transformers,
diffusers, timm. It does not force unrelated models through the selected library.
`--library-name` is required with `--framework optimum`, including estimate runs.
Add `--estimate` to count the selected models without exporting them.
Add `--verbose` to show Optimum worker output, including model download progress
and backend warnings. Worker output is suppressed by default; this option does
not change the output of other conversion frameworks.

For TensorFlow, install `scripts-hubs-hf-tensorflow` in a Python 3.11 environment
and use `--framework keras` or `--framework tflite`, without `--library-name`.
For Stable-Baselines3, use its extra and `--framework stable_baselines3`.
`--library-name` is only valid with `--framework optimum`.

Use a separate `--cache-dirpath` for each environment/run: conversion clears its
working cache. Optimum status and resume files include the selected library name;
other frameworks retain their original status filenames. Existing
combined-run checkpoints are not reused by a library-specific run.

For PyPI installations, Python 3.11 support requires releases of both the kernel
and IR with the updated Python metadata. Until then, use the source installation
shown above. Install the matching published extra once those releases are available.

## Prepare Hugging Face conversion jobs

Read all `*.json` files directly inside a Model_Infos directory, filter by
`lastModified`, and sort each conversion group in ascending date order before
splitting. Use an empty output directory:

```bash
mkdir -p jobs-2026
younger-logics-ir create onnx prepare huggingface \
    --model-infos-dirpath ./model-infos \
    --save-dirpath ./jobs-2026 \
    --date-l 2026-01-01 \
    --date-r 2027-01-01 \
    --number-per-file 1000
```

Dates must use `YYYY-MM-DD` and represent 00:00 UTC on that day. `date_l` is
inclusive and `date_r` is exclusive; either may be omitted for an open bound.
Equal dates are ordered by model ID. Identical duplicate records are deduplicated;
conflicting records for the same ID are rejected. Input JSON files are loaded one at a time.
SQLite stores the full dataset for deduplication and sorting; output records are
loaded and saved one shard at a time. `--number-per-file` is required and must be
positive. Memory use depends on the largest input file, output shard size and
job manifest, rather than all model records. Reserve disk space for the database
and output files; the temporary database is removed when preparation exits.
The input directory should contain only Model_Infos JSON lists.

Shard filenames contain framework, library name, date bounds
in `YYYY-MM-DD` format, and a zero-based index, for example:

```text
optimum_diffusers_2026-01-01_2027-01-01_00000.json
optimum_diffusers_2026-02-03_2026-11-04_00000.json
```

The library name is omitted when absent. An omitted date bound is named using
the minimum or maximum valid UTC `lastModified` date across the input directory.
These inferred dates describe the input data and do not impose additional filters;
models on the maximum date are included. Explicit `date_r` remains exclusive.
For example, explicit bounds 2026-01-01 to 2027-01-01 cover all of 2026.
Shards are written directly
into the output directory. `jobs.json` records their paths, frameworks, libraries
and counts. Each shard preserves the original model records and can be passed to
conversion's `--model-infos-filepath`. Use a separate conversion cache directory
for every shard/server to keep resume files independent.

No framework or library input is needed: all groups identified by the existing
conversion tag/file rules are output. Models supporting several
file formats can appear in several framework groups. Unrecognized routes use
`unknown` in the filename; missing or invalid dates are saved in
`missing_dates_00000.json`, etc., outside the job list, using the same shard
size limit. Their filenames are recorded in `jobs.json`. These records require inspection.
Routing does not guarantee successful export. `--number-per-file` limits model
count, not conversion time or memory.

## Hugging Face conversion output

Instances are stored by model, conversion framework and requested ONNX opset:

```text
Instances/<owner>/<model>/
    README.md
    <framework>/opset_<version>/<instance_unique>/
        meta
        logicx.bin
        labels.lbl
        conversion.json
```

`conversion.json` records the model ID, framework, library, artifact, requested
`opset_version`, and actual ONNX `opset_import` by domain. The latter is also
stored in the LogicX graph. Requested and actual versions are recorded separately
because exporters may select a different opset. README content is saved as Markdown
once per model. Dataset traversal must descend into the model/framework/opset
folders to find Instance directories. Existing outputs and resume records are
not migrated; use a fresh conversion cache when regenerating the new layout.
