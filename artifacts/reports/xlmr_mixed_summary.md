# XLM-RoBERTa run: xlmr_mixed

- Base model: `xlm-roberta-base`, seed 42, smoke run: False
- Training rows: 45,470 (label counts {'negative': 15250, 'neutral': 14965, 'positive': 15255})
- Best checkpoint metric (validation macro-F1): 0.8872361167137287
- Training time: 553s

| Dataset | Role | Rows | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: |
| natural | validation (checkpoint selection) | 442 | 0.8872 | 0.8869 |
| synthetic | report only | 15,000 | 0.9998 | 0.9998 |

The test split was not used.
