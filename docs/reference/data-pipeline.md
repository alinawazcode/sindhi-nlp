# Data pipeline

Two entry points build the files that training and evaluation read. Both are deterministic (seeded) and write a report with counts and file hashes.

## Human-labeled data: `prepare_external`

```bash
python -m sindhi_nlp.data.prepare_external
    [--config configs/base.yaml] [--external-config configs/external.yaml] [--project-root .]
```

Input: the CSV named in `configs/external.yaml`. Steps:

1. read all columns as text (no type guessing);
2. rename your columns to `Text` and `Label`;
3. clean the text ([options](../guide/preprocessing.md));
4. normalize labels (case-insensitive, aliases) and drop rows with missing text or unknown labels, counting each;
5. resolve duplicates according to `duplicate_policy`;
6. split into train, validation and test (stratified, grouped, seeded);
7. check for overlap between splits, and against the files in `compare_with`;
8. write `train.csv`, `validation.csv`, `test.csv` and `split_report.json`.

It refuses a source file inside `data/raw/` and stops with a clear error if the file is missing. Files must be UTF-8 (a byte-order mark is fine); re-save Excel exports as *CSV UTF-8*.

## Synthetic data: `split_data`

```bash
python -m sindhi_nlp.data.split_data --config configs/base.yaml [--project-root .]
```

Reads every CSV in `data/raw/`, merges and cleans them, validates, removes duplicates, builds grouped stratified splits and writes `data/interim/merged_clean.csv`, `data/processed/*.csv`, `label_mapping.json` and `data/reports/split_report.json`. The synthetic data is not part of the repository; this step is only needed to reproduce the synthetic experiments.

## The split report

`split_report.json` records: row counts at each stage, labels per split, how many rows were dropped and why, duplicate counts, the seed, the cleaning options, SHA-256 hashes of input and output files, and library versions. Two runs on the same input should give identical hashes. The small reports are committed so the numbers in this documentation can be checked.

## Functions

| Module | Function | Description |
| --- | --- | --- |
| `data.load_data` | `read_csv`, `load_raw_directory`, `load_processed_split`, `file_sha256` | read UTF-8 files (a byte-order mark is tolerated) as plain strings, with no type guessing |
| `data.clean_data` | `clean_text`, `clean_series`, `CleaningOptions` | normalization |
| `data.validate_data` | `validate_dataframe`, `normalize_label`, `filter_valid_rows`, `resolve_duplicates`, `ValidationReport` | schema, labels, duplicates |
| `data.split_data` | `grouped_stratified_split`, `make_group_key_fn`, `assert_disjoint`, `text_overlap`, `group_overlap`, `write_csv`, `write_json` | splitting and safe writing |

## Common problems

| Message | Fix |
| --- | --- |
| `DataValidationError` about columns | set `text_column` / `label_column` in `external.yaml` (or `raw_text_columns` in `base.yaml`) |
| fewer rows than expected | check the report: rows with unknown labels, missing text and duplicates are dropped and counted |
| `ConfigError` | a required key is missing; the message names it |
| Windows: garbled text in the config | save the YAML as UTF-8 |
