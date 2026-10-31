# FAQ and troubleshooting

## General

**Which languages and scripts?** Sindhi in Arabic script. Other scripts (Devanagari Sindhi), transliterated Sindhi and mixed-language text are untested.

**Can I use it commercially?** The code and model weights are MIT licensed; keep the author credit. The 4,420-sentence dataset is separate and is licensed separately under CC BY-SA 4.0 (attribution and share-alike).

**Can I use it for moderation, hiring or credit decisions?** No, not without human review. See the [model card](model_card.md).

**How accurate is it on my data?** Unknown until you measure. The published score is 0.891 macro-F1 on one news-style test set. See [Evaluate on your own data](tutorials/evaluate-own-data.md).

**Is the neutral class "no sentiment"?** No. In the training data, neutral means normal, everyday conversation without a clearly positive or negative feeling, as judged by the one annotator. The model may partly use topic to decide.

## Installation and loading

**`ModuleNotFoundError: No module named 'torch'`** You loaded the XLM-R model without the extra. Run `pip install "sindhi-nlp[transformers]"`, or use the baseline.

**`Loading a model by name needs: pip install huggingface_hub`** Install `sindhi-nlp[hub]` (the `transformers` extra includes it).

**What are the `Fetching 8 files`, `Loading weights` and `Reconstruction complete` lines?** Progress messages from the Hugging Face libraries when the model loads. The results are printed after them. To hide the download bars, set `HF_HUB_DISABLE_PROGRESS_BARS=1` (PowerShell: `$env:HF_HUB_DISABLE_PROGRESS_BARS="1"`).

**The first prediction takes minutes.** The model (about 1.1 GB) is downloading. It is cached afterwards.

**`ModelNotFoundError: ... could not be downloaded from Hugging Face`** Check the name for typos, your internet connection, and any proxy settings. A relative local path must exist from your current folder.

**I want to run offline.** Download the model folder once, copy it, and pass the folder path. Setting `HF_HUB_OFFLINE=1` makes the Hugging Face libraries use only the local cache.

**`sindhi-nlp` is not recognized (Windows).** The scripts folder is not on `PATH`. Use `python -m sindhi_nlp ...`, which does the same.

## Using it

**Why does `predict_batch` raise for one bad row?** The default is `errors="raise"`. Use `errors="none"` to get `None` for bad rows.

**Why is confidence always close to 1.0?** XLM-R is overconfident. See [Confidence scores](concepts/confidence.md).

**Sindhi text looks garbled in my terminal.** The console encoding is not UTF-8. Use a file (`batch`), or in PowerShell set `[Console]::OutputEncoding = [Text.Encoding]::UTF8`.

**Excel shows garbled text in the output CSV.** Output files are UTF-8 with a BOM, which Excel reads correctly. If you saved the file again from another program, re-save it as CSV UTF-8.

**Texts longer than 128 tokens.** They are truncated for the published model; only the start of a long text is read. Split long documents into sentences first.

**Is it fast enough?** XLM-R on a CPU handles a few sentences per second; the baseline is far faster. A GPU speeds XLM-R up considerably.

**Is the predictor thread-safe?** Do not share one predictor between threads without a lock. The HTTP API serializes access for you.

## Evaluation and training

**`The test split is a one-time final evaluation`** Add `--final`, once you are done choosing models. This is a safety rule, not a bug.

**`already exists: the test split was already scored`** Same rule. Use `--overwrite` only if you accept that the test set is no longer unseen.

**Why is the baseline so close to XLM-R?** The task as labeled is largely solvable from vocabulary. The gap is within the noise of the test set (p = 0.116).

**Can I train XLM-R on a CPU?** Technically yes, but it is impractically slow. Use a free GPU service such as Kaggle, then copy the model folder back.

## API

**Why does `/health` return 503?** The model failed to load; the server log says why.

**Why are requests slow under load?** One request is served at a time; the model is the bottleneck. Use the baseline, a GPU, or several instances behind a gateway.

## Still stuck?

Open an issue at <https://github.com/alinawazcode/sindhi-nlp/issues> with the output of `sindhi-nlp --version`, your Python version and the full error message. Do not paste private data.
