# Publishing

## 1. Hugging Face model

The upload tool is a dry run unless you tell it otherwise. It checks the model folder, builds the repository's README from `docs/model_card.md`, and lists exactly what would be uploaded.

One-time setup:

```powershell
pip install huggingface_hub
hf auth login          # paste a token with "write" access from https://huggingface.co/settings/tokens
sindhi-nlp finalize-model --model-dir artifacts/models/xlmr_mixed    # writes preprocessing.json
```

Dry run (changes nothing):

```powershell
python -m sindhi_nlp.cli.publish_model --model-dir artifacts/models/xlmr_mixed
```

It uploads only the weights, tokenizer, configuration, label mapping, preprocessing settings, training configuration, and metrics. It leaves out `training_args.bin` (a pickle file), the training log, and checkpoints. It refuses to continue if any file would publish a path from your computer, such as `C:\Users\...`.

Upload as a **private** repository first and look at it on the website:

```powershell
python -m sindhi_nlp.cli.publish_model --model-dir artifacts/models/xlmr_mixed --upload
```

When it looks right, make it public. The tool asks you to type the repository name to confirm:

```powershell
python -m sindhi_nlp.cli.publish_model --model-dir artifacts/models/xlmr_mixed --upload --public
```

Check the published model from a clean folder:

```powershell
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" --model-dir alinawazmahar/sindhi-sentiment-xlmr
```

The first call downloads the weights (about 1.1 GB) into the Hugging Face cache. Later calls reuse them. To serve the API straight from the Hub, set `model_dir: alinawazmahar/sindhi-sentiment-xlmr` in `configs/inference.yaml`; use `revision` to pin a tag or commit.

Publishing is not easily undone: files remain in the repository's history, and once a public model has been downloaded it cannot be recalled.

## 2. PyPI package

A version number on PyPI can never be reused, even if the release is deleted, so rehearse on TestPyPI first. TestPyPI is a separate site with a separate account and token.

Never paste a token into a file, a commit, or a chat. Set it in the current terminal session only.

Build and check (from a clean `git status`, on the commit you want to release):

```powershell
pip install build twine
Remove-Item dist, build -Recurse -Force -ErrorAction SilentlyContinue
python -m build
twine check dist/*
```

Rehearsal on TestPyPI:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<your TestPyPI token, starts with pypi->"
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Install it into a fresh environment and try it. The extra index lets pip fetch dependencies from the real PyPI:

```powershell
python -m venv $env:TEMP\tp_venv
& $env:TEMP\tp_venv\Scripts\python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "sindhi-nlp[transformers]"
& $env:TEMP\tp_venv\Scripts\sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" --model-dir alinawazmahar/sindhi-sentiment-xlmr
```

The real release:

```powershell
$env:TWINE_PASSWORD = "<your PyPI token, starts with pypi->"
twine upload dist/*
git tag v0.1.0
git push origin v0.1.0
```

If a release has a mistake, do not try to replace it: fix the problem and publish the next version number (for example 0.1.3), updating `__version__` in `src/sindhi_nlp/__init__.py`.
