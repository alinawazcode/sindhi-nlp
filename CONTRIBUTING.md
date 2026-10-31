# Contributing

Thanks for helping. Bug reports, documentation fixes, evaluations on new domains, and carefully licensed labeled data are all welcome.

## Ground rules

- **Never commit** datasets, model weights, checkpoints, tokens or `.env` files. The `.gitignore` blocks most of them; check `git status` before every commit.
- **Never look at the test split to make decisions.** Choices are made on validation data. The test split is scored once per model (`--final`).
- **Changed behavior needs a test**, and a line in `CHANGELOG.md` under "Unreleased".
- Be honest in results. If a gain is not statistically significant, say so.

## Set up

```bash
git clone https://github.com/alinawazcode/sindhi-nlp.git
cd sindhi-nlp
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[api,plots,dev,docs]"
pip install -e ".[transformers]"                        # only if you work on XLM-R code
pre-commit install                                      # optional: runs the checks on commit
```

## Before you open a pull request

```bash
ruff check .
ruff format --check .      # `ruff format .` fixes formatting
mypy
pytest
mkdocs build --strict      # if you changed documentation
```

All of these run in CI on Python 3.11 and 3.12. Tests that need PyTorch or FastAPI are skipped when those are not installed, so install the extras to run everything.

## Code style

- Python 3.11+, type hints on public functions, line length 120 (`ruff` enforces the rest).
- Public functions and classes get a docstring: one summary line, then `Args`, `Returns`, `Raises` and an example when it helps. Examples in docstrings should run.
- Keep repeatable work in `src/`; notebooks are for exploration only.
- Small, focused pull requests are easier to review than large ones.

## Documentation

The site is built with MkDocs Material from `docs/` and `mkdocs.yml`.

```bash
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build --strict
```

Rules for documentation: show real commands and real output; mark anything illustrative as such; never invent results; update the page when you change the behavior it describes.

## Reporting bugs

Open an issue with: `sindhi-nlp --version`, Python version and OS, what you ran, what you expected, and the full error. Please do not include private data. Report security problems privately, as described in [SECURITY.md](https://github.com/alinawazcode/sindhi-nlp/blob/main/SECURITY.md).

## Contributing data

Labeled sentences are valuable, but only if their license allows redistribution. Say where the text came from, who labeled it, how, and under what license. Do not send text you do not have the right to share.

## License

By contributing you agree that your contribution is released under the MIT License.
