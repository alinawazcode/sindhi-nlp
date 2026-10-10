# XLM-RoBERTa run: xlmr_natural

- Base model: `xlm-roberta-base`, seed 42, smoke run: False
- Training rows: 3,094 (label counts {'negative': 1050, 'neutral': 993, 'positive': 1051})
- Best checkpoint metric (validation macro-F1): 0.8662705094855716
- Training time: 130s

| Dataset | Role | Rows | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: |
| natural | validation (checkpoint selection) | 442 | 0.8663 | 0.8665 |
| synthetic | report only | 15,000 | 0.8295 | 0.8266 |

The test split was not used.
