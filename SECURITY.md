# Security policy

## Supported versions

Security fixes are made to the latest release.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Email alinawaz.code@gmail.com with a description, steps to reproduce and the affected version. You can expect an acknowledgement within a week.

## What is in scope

- the Python package and command line (`sindhi-nlp` on PyPI);
- the HTTP service in `sindhi_nlp.api`, including input handling;
- the model upload tool, which must never publish secrets or local paths.

## Notes for users

- The HTTP API has **no authentication or rate limiting**. Do not expose it directly to the internet.
- Models are loaded from safetensors or joblib files. A `joblib` file is a pickle: **only load model folders you trust.** The Hugging Face download is restricted to `.json`, `.safetensors`, `.model` and `.txt` files, so it never downloads pickle weights.
- Never commit API tokens. If you pasted one anywhere public, revoke it immediately at the provider and create a new one, preferably restricted to a single project.
