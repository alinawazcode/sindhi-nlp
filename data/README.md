# Data folder

Datasets are not stored in Git. See [docs/dataset.md](../docs/dataset.md) for what each dataset contains, where it came from, how it is split, and its known limitations.

| Folder | What goes here |
| --- | --- |
| `raw/` | The original synthetic CSV files. Never modify them after placing them here. |
| `interim/` | Cleaned and merged data generated during preparation. |
| `processed/` | `train.csv`, `validation.csv`, `test.csv`, and `label_mapping.json` for the synthetic data. |
| `external/` | `natural_sindhi.csv`, the human-labeled dataset (not the same data as `raw/`). Its prepared splits go in `external/processed/`. |
| `reports/` | `split_report.json` with row counts, seed, and file hashes. |

Rebuild the processed files with the commands in the README.
