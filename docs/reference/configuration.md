# Configuration files

All settings live in YAML files under `configs/`. Paths inside them are relative to the project root (the folder you run commands from, or `--project-root`). Save the files as UTF-8: some settings contain Arabic-script text.

| File | Used by |
| --- | --- |
| `base.yaml` | every stage: seed, labels, columns, paths, split, cleaning |
| `external.yaml` | preparing the human-labeled dataset |
| `baseline.yaml` | TF-IDF + LinearSVC training |
| `xlm_roberta_base.yaml` | XLM-R fine-tuning (general defaults: 4 epochs) |
| `xlmr_mixed.yaml` | the exact settings of the published model (3 epochs, 30,000 synthetic rows, human-labeled data repeated 5 times) |
| `inference.yaml` | the CLI, `SentimentPredictor.from_config` and the HTTP API |

## base.yaml {#base-yaml}

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | non-negative integer; seeds Python, NumPy, PyTorch and the splits |
| `labels` | `[negative, neutral, positive]` | label order (the confusion-matrix order) |
| `label_mapping` | `negative: 0, neutral: 1, positive: 2` | class ids |
| `columns.text`, `columns.label` | `Text`, `Label` | column names in the processed files |
| `paths.*` | `data/raw`, `data/interim`, `data/processed`, `data/reports`, `artifacts/models`, `artifacts/metrics`, `artifacts/reports` | where things are read and written |
| `data.split.train/validation/test` | `0.8 / 0.1 / 0.1` | split ratios (synthetic data) |
| `data.duplicate_policy` | `drop` | `drop` removes conflicting-label texts and keeps the first copy of same-label duplicates; `report` only counts them |
| `data.raw_text_columns`, `raw_label_columns` | `[]` | extra accepted header names in raw files |
| `data.label_aliases` | `{}` | map raw spellings to labels, e.g. `{pos: positive}` |
| `data.grouping.enabled` | `true` | keep near-duplicates in one split |
| `data.grouping.ignore_punctuation_digits` | `false` | also ignore punctuation and digits when grouping |
| `data.grouping.normalizers` | weekday names to `<DAY>` | ordered `pattern` / `replace` regex rules that build the group key |
| `data.cleaning.*` | see [preprocessing](../guide/preprocessing.md#options) | text cleaning options |

## external.yaml {#external-yaml}

| Key | Meaning |
| --- | --- |
| `source_file` | your labeled CSV (never inside `data/raw/`) |
| `text_column`, `label_column` | header names in your file, matched case-insensitively |
| `duplicate_policy` | `drop` or `report` |
| `split.train/validation/test` | ratios; the default is `0.7 / 0.1 / 0.2` |
| `output_dir` | where the prepared files go |
| `compare_with` | other split files to check for text overlap (contamination) |

## baseline.yaml {#baseline-yaml}

| Key | Default | Meaning |
| --- | --- | --- |
| `model_name` | `baseline_tfidf_linearsvc` | output name (overridden by `--model-name`) |
| `tfidf.analyzer` | `word` | features |
| `tfidf.token_pattern` | word characters plus Arabic combining marks | what counts as a word |
| `tfidf.ngram_range` | `[1, 2]` | unigrams and bigrams |
| `tfidf.min_df` | `2` | drop words seen in fewer documents |
| `tfidf.max_features` | `300000` | vocabulary cap |
| `tfidf.sublinear_tf` | `true` | log-scaled term frequency |
| `tfidf.lowercase` | `true` | |
| `svc.C` | `1.0` | regularization strength (larger = weaker regularization) |
| `svc.class_weight` | `null` | `balanced` to reweight classes |
| `svc.max_iter` | `5000` | solver iterations |
| `analysis.top_features_per_class` | `15` | words listed in the summary |
| `analysis.max_error_examples` | `200` | mistakes saved |

The baseline is deliberately simple. It is the comparison point for XLM-R, so avoid heavy tuning on the validation split unless you compare fairly.

## xlm_roberta_base.yaml {#xlm-roberta-base-yaml}

| Key | Default | Meaning |
| --- | --- | --- |
| `model_name` | `xlm-roberta-base` | the pretrained model to fine-tune |
| `run_name` | `xlmr_base` | output folder name under `artifacts/models/` |
| `datasets` | `synthetic`, `natural` | name to folder (each needs `train.csv` and `validation.csv`) |
| `train_on` | `[synthetic, natural]` | datasets combined for training |
| `validate_on` | `[natural]` | validation splits used to choose the checkpoint |
| `report_on` | `[synthetic]` | extra validation splits, reported but never used to choose |
| `max_rows` | `{}` | per-dataset training cap (stratified), e.g. `{synthetic: 20000}` |
| `repeat` | `{}` | per-dataset repetition factor, e.g. `{natural: 5}` |
| `tokenizer.max_length` | `128` | token limit; the run logs token-length percentiles so you can check it |
| `training.learning_rate` | `2.0e-5` | |
| `training.per_device_train_batch_size` | `32` | |
| `training.per_device_eval_batch_size` | `64` | |
| `training.gradient_accumulation_steps` | `1` | raise to simulate a larger batch with less memory |
| `training.num_train_epochs` | `4` | upper bound; early stopping can end sooner |
| `training.warmup_ratio` | `0.1` | |
| `training.weight_decay` | `0.01` | |
| `training.early_stopping_patience` | `2` | evaluations without improvement before stopping; `0` disables |
| `training.save_total_limit` | `2` | checkpoints kept during training |
| `training.logging_steps` | `50` | |
| `training.dataloader_num_workers` | `0` | |
| `training.fp16` | `auto` | on when a CUDA GPU is available |
| `smoke.*` | tiny sizes | the settings used by `--smoke` |

`repeat` and `max_rows` may name only datasets in `train_on`. A config problem raises `ConfigError` before any training starts.

## inference.yaml {#inference-yaml}

| Key | Default | Meaning |
| --- | --- | --- |
| `model_dir` | `artifacts/models/xlmr_mixed` | local folder, or a Hugging Face name; required |
| `revision` | not set | Hub revision (branch, tag, commit) |
| `device` | `auto` | `auto`, `cpu` or `cuda` |
| `batch_size` | `32` | texts per forward pass |
| `max_length` | `null` | token limit; `null` uses the value saved with the model |
| `max_chars` | `10000` | longer inputs are rejected before tokenizing |
| `confidence_threshold` | `null` | e.g. `0.6`; results below it get `is_confident: false` |
| `api.max_batch_size` | `64` | largest `/predict/batch` request |
| `api.cors_origins` | `[]` | browser origins allowed to call the API |

Environment variables for the API: `SINDHI_NLP_CONFIG` (path of the config file) and `SINDHI_NLP_ROOT` (project folder).
