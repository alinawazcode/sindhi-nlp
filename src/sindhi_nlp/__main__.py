"""Allow ``python -m sindhi_nlp`` as an alias for the ``sindhi-nlp`` command."""

from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
