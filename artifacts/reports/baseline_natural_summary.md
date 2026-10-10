# Baseline report: baseline_natural

Trained on `train.csv`, evaluated on `validation.csv`. The test split was not used.

## Validation results

- Macro-F1: **0.8665**
- Accuracy: 0.8665
- Macro precision / recall: 0.8677 / 0.8661
- Train time: 0.3s, validation prediction: 0.03s

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8599 | 0.9000 | 0.8795 | 150 |
| neutral | 0.8955 | 0.8451 | 0.8696 | 142 |
| positive | 0.8477 | 0.8533 | 0.8505 | 150 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 135 | 4 | 11 |
| neutral | 10 | 120 | 12 |
| positive | 12 | 10 | 128 |

## Most common mistakes

- neutral predicted as positive: 12
- positive predicted as negative: 12
- negative predicted as positive: 11
- neutral predicted as negative: 10
- positive predicted as neutral: 10
- negative predicted as neutral: 4

## Highest-weighted terms per class

- **negative**: نه, خراب, پريشان, ناانصافي, سخت, نقصان, ڪاوڙ, غلط, اونداهي, بيروزگاري, کوٽ, پريشاني, هيءَ, انتهائي, ناهي
- **neutral**: پاڪستان, شروع, اهي, ڊاڪٽر, بازار, شهر, بجيٽ, اعلان, خاموش, اسپتال, پاڪستاني, ڪتاب, ڪاروباري, هو, ڪاليج
- **positive**: پنهنجي, خوشي, خوش, خوشگوار, حاصل, پيار, ڪامياب, پهريون, خير, محبت, سان, سفر, مدد, روح, آخرڪار

## Setup

- Seed: 42
- TF-IDF: `{"analyzer": "word", "token_pattern": "(?u)[\\w\\u064B-\\u065F\\u0670]+", "ngram_range": [1, 2], "min_df": 2, "max_features": 300000, "sublinear_tf": true, "lowercase": true}`
- LinearSVC: `{"C": 1.0, "class_weight": null, "max_iter": 5000}`
- Train rows: 3,094, validation rows: 442
