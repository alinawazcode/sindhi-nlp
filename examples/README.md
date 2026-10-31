# Examples

Complete, runnable scripts. Each one starts with a comment saying what it does and how to run it. They are also shown, with explanations, on the [Examples page](https://alinawazcode.github.io/sindhi-nlp/examples/) of the documentation.

```bash
pip install "sindhi-nlp[transformers]"
python examples/01_classify_text.py
```

| Script | What it shows |
| --- | --- |
| `01_classify_text.py` | one sentence, many sentences, model details |
| `02_classify_csv.py` | score a CSV file and list the rows to review |
| `03_handle_bad_input.py` | empty or invalid text, and confidence thresholds |
| `04_serve_and_call_api.py` | call the HTTP API from Python |
| `05_evaluate_own_data.py` | measure the model on your own labeled sentences |

The first run downloads the published model (about 1.1 GB) and caches it. To try the scripts with another model, set `SINDHI_MODEL` to a local folder or a Hugging Face name.
