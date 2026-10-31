# Installation

Requires **Python 3.11 or newer**. The test suite runs on Linux in CI. The published model was also run by the author on Windows with Python 3.12 on a CPU, using PyTorch 2.13 and Transformers 5.17. macOS is expected to work but has not been tested.

## Choose what you need

| Command | Installs | Use it for |
| --- | --- | --- |
| `pip install sindhi-nlp` | core: numpy, pandas, scikit-learn, scipy, joblib, PyYAML | the TF-IDF baseline, data tools, CLI |
| `pip install "sindhi-nlp[transformers]"` | adds PyTorch and Hugging Face libraries | the XLM-R model (recommended) |
| `pip install "sindhi-nlp[hub]"` | adds `huggingface_hub` | loading a model by Hugging Face name without PyTorch (the `transformers` extra already brings it) |
| `pip install "sindhi-nlp[api]"` | adds FastAPI and uvicorn | the HTTP API |
| `pip install "sindhi-nlp[plots]"` | adds matplotlib | confusion-matrix and comparison figures |
| `pip install "sindhi-nlp[dev]"` | test and lint tools | working on the code |
| `pip install "sindhi-nlp[docs]"` | mkdocs and the Material theme | building this site |

Extras combine: `pip install "sindhi-nlp[transformers,api,plots]"`.

!!! note "Check the extras in your version"
    The exact extras and pinned minimum versions are defined in [`pyproject.toml`](https://github.com/alinawazcode/sindhi-nlp/blob/main/pyproject.toml). `pip show sindhi-nlp` tells you which version you have, and `sindhi-nlp --version` prints it.

## Use a virtual environment

=== "Windows PowerShell"

    ```powershell
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install "sindhi-nlp[transformers]"
    ```

=== "Linux / macOS"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install "sindhi-nlp[transformers]"
    ```

## First run downloads the model

The first time you load `alinawazmahar/sindhi-sentiment-xlmr`, the library downloads about 1.1 GB from the Hugging Face Hub and caches it, so later runs start immediately. To choose where it is cached, pass `cache_dir=` to `from_pretrained`, or set the standard `HF_HOME` environment variable.

For machines without internet access, download the model folder once on a connected machine, copy it across, and pass the local folder path instead of the name.

## GPU or CPU

Nothing extra is needed for a CPU. XLM-R on a CPU classifies a few sentences per second, which is fine for scripts and small batches. If you have a CUDA GPU and a CUDA build of PyTorch, the default `device="auto"` uses it. Install the PyTorch build that matches your hardware first, following <https://pytorch.org/get-started/locally/>, then install `sindhi-nlp[transformers]`.

## Install from source

```bash
git clone https://github.com/alinawazcode/sindhi-nlp.git
cd sindhi-nlp
pip install -e ".[transformers,api,plots,dev]"
```

## Verify the installation

```bash
sindhi-nlp --version
python -c "import sindhi_nlp; print(sindhi_nlp.__version__)"
```

Both print the same version. (`--version` was added in 0.1.2; with 0.1.1 or earlier the command prints a usage error instead. See [Which version has what](#which-version-has-what).) To check that a model loads and runs:

```bash
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" --model-dir alinawazmahar/sindhi-sentiment-xlmr
```

## Which version has what

Version 0.1.2 is the latest release on PyPI. It added the following to 0.1.1:

| Added in 0.1.2 | Where it is described |
| --- | --- |
| `python -m sindhi_nlp.evaluation.evaluate`, `error_analysis`, `plots` | [Evaluation tools](reference/evaluation.md) |
| `sindhi-nlp --version`, `python -m sindhi_nlp` | [Command line](guide/cli.md) |
| `plots` and `docs` extras, `py.typed` | this page |
| `configs/xlmr_mixed.yaml` (reproduces the published model) | [Configuration files](reference/configuration.md) |
| Clean `Error:` message from `evaluate_external` instead of a traceback | [Errors](reference/errors.md) |
| `finalize-model` works after `pip install` (no `configs/` folder needed) | [Command line](guide/cli.md) |

Everything else in this documentation, including the SDK, the CLI commands `predict`, `batch`, `info` and `finalize-model`, and the HTTP API, is in 0.1.1 and 0.1.2. To get the additions, upgrade:

```bash
pip install --upgrade sindhi-nlp
```

The `configs/` folder (used by training, data preparation and `compare_models`) is part of the GitHub repository and is not installed by pip. Clone the repository to use those tools.

## Upgrade or uninstall

```bash
pip install --upgrade sindhi-nlp
pip uninstall sindhi-nlp
```

The cached model stays in the Hugging Face cache until you delete it.
