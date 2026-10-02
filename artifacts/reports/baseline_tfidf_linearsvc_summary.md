# Baseline report: baseline_tfidf_linearsvc

Trained on `train.csv`, evaluated on `validation.csv`. The test split was not used.

## Validation results

- Macro-F1: **0.9999**
- Accuracy: 0.9999
- Macro precision / recall: 0.9999 / 0.9999
- Train time: 15.2s, validation prediction: 1.29s

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.9998 | 1.0000 | 0.9999 | 5,000 |
| neutral | 0.9998 | 1.0000 | 0.9999 | 5,000 |
| positive | 1.0000 | 0.9996 | 0.9998 | 5,000 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 5,000 | 0 | 0 |
| neutral | 0 | 5,000 | 0 |
| positive | 1 | 1 | 4,998 |

## Most common mistakes

- positive predicted as negative: 1
- positive predicted as neutral: 1

## Highest-weighted terms per class

- **negative**: خراب, نه, پريشان, مايوس, تڪليف, مايوسي, غير, پريشاني, تمام پريشان, بلڪل, پريشان رهي, ضايع, ڏک, چوري ٿي, تمام خراب
- **neutral**: ڏيکاريو, جاري, رهي ٿي, عام, اڪائونٽ آهي, ويو, گهربل هو, گهربل, آهي, پهتو, گهر هو, پيو وڃي, اڃا, ڳالهايو, سان ڳالهايو
- **positive**: خوش, تمام, سٺو, سٺي, شاندار, بهترين, جلدي, سستي اگهه, اگهه خريد, تعريف, سستي, خوشي, تمام پسند, خوشيءَ, يادگار

## Setup

- Seed: 42
- TF-IDF: `{"analyzer": "word", "token_pattern": "(?u)[\\w\\u064B-\\u065F\\u0670]+", "ngram_range": [1, 2], "min_df": 2, "max_features": 300000, "sublinear_tf": true, "lowercase": true}`
- LinearSVC: `{"C": 1.0, "class_weight": null, "max_iter": 5000}`
- Train rows: 120,000, validation rows: 15,000
