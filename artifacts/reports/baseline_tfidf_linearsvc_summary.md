# Baseline report: baseline_tfidf_linearsvc

Trained on `train.csv`, evaluated on `validation.csv`. The test split was not used.

## Validation results

- Macro-F1: **0.9998**
- Accuracy: 0.9998
- Macro precision / recall: 0.9998 / 0.9998
- Train time: 9.7s, validation prediction: 0.51s

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.9998 | 0.9996 | 0.9997 | 5,000 |
| neutral | 0.9998 | 1.0000 | 0.9999 | 5,000 |
| positive | 0.9998 | 0.9998 | 0.9998 | 5,000 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 4,998 | 1 | 1 |
| neutral | 0 | 5,000 | 0 |
| positive | 1 | 0 | 4,999 |

## Most common mistakes

- negative predicted as neutral: 1
- negative predicted as positive: 1
- positive predicted as negative: 1

## Highest-weighted terms per class

- **negative**: خراب, نه, پريشان, مايوس, تڪليف, بلڪل, غير, پريشاني, مايوسي, ضايع, تمام پريشان, پريشان رهي, ڏک, چوري ٿي, تمام خراب
- **neutral**: ڏيکاريو, جاري, رهي ٿي, عام, اڪائونٽ آهي, ويو, آهي, گهربل, گهربل هو, پهتو, گهر هو, اڃا, پيو وڃي, سان ڳالهايو, ڳالهايو
- **positive**: خوش, تمام, سٺو, سٺي, بهترين, شاندار, جلدي, سستي اگهه, اگهه خريد, تعريف, خوشي, تمام پسند, تمام سٺي, يادگار, سستي

## Setup

- Seed: 42
- TF-IDF: `{"analyzer": "word", "token_pattern": "(?u)[\\w\\u064B-\\u065F\\u0670]+", "ngram_range": [1, 2], "min_df": 2, "max_features": 300000, "sublinear_tf": true, "lowercase": true}`
- LinearSVC: `{"C": 1.0, "class_weight": null, "max_iter": 5000}`
- Train rows: 120,000, validation rows: 15,000
