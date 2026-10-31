# Examples

Complete scripts you can copy, save and run. They live in the [`examples/`](https://github.com/alinawazcode/sindhi-nlp/tree/main/examples) folder of the repository and are shown here exactly as they are in the repo.

Install once:

```bash
pip install "sindhi-nlp[transformers]"
```

The first run downloads the published model (about 1.1 GB) and caches it. Every script loads the model through this small helper, so you can point all of them at another model by setting the `SINDHI_MODEL` environment variable:

```python title="_model.py"
--8 < --"examples/_model.py"
```

Save `_model.py` next to the script you want to run.

## 1. Classify text

One sentence, several sentences, and details of the model.

```python title="01_classify_text.py"
--8 < --"examples/01_classify_text.py"
```

## 2. Classify a CSV file

Reads a UTF-8 CSV with a `Text` column, writes the same file with `predicted_label` and `confidence` added, and prints the rows a person should check. Blank rows do not stop the run.

```python title="02_classify_csv.py"
--8 < --"examples/02_classify_csv.py"
```

```bash
python 02_classify_csv.py reviews.csv reviews_scored.csv
```

The same job without writing code: `sindhi-nlp batch --input reviews.csv --output reviews_scored.csv` ([tutorial](tutorials/classify-csv.md)).

## 3. Bad input and confidence thresholds

```python title="03_handle_bad_input.py"
--8 < --"examples/03_handle_bad_input.py"
```

## 4. Call the HTTP API

Start the server first ([Serve the API](tutorials/serve-api.md)), then:

```python title="04_serve_and_call_api.py"
--8 < --"examples/04_serve_and_call_api.py"
```

## 5. Evaluate on your own data

```python title="05_evaluate_own_data.py"
--8 < --"examples/05_evaluate_own_data.py"
```

See [Evaluate on your own data](tutorials/evaluate-own-data.md) for how to build a labeled file and read the numbers.

## From the command line

```bash
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي"
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" "مون کي ڏاڍو ڏک ٿيو" --json
sindhi-nlp batch --input reviews.csv --output reviews_scored.csv
sindhi-nlp info
```

All commands and options are on the [Command line](guide/cli.md) page.
