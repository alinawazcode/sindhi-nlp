# Model card: Sindhi sentiment classifier


## Model details

| | |
| --- | --- |
| Task | Sentence-level sentiment classification of Sindhi text: `negative`, `neutral`, `positive` |
| Main model | `xlmr_mixed`: `xlm-roberta-base` fine-tuned for sequence classification |
| Reference model | `baseline_natural`: TF-IDF (word 1-2 grams) + LinearSVC |
| Framework | PyTorch, Hugging Face Transformers; scikit-learn for the baseline |
| Seed | 42 |
| Author | Ali Nawaz |
| Contact | alinawaz.code@gmail.com |
| License | Model weights are free to use under the MIT License, the same license as the base model `xlm-roberta-base`. The natural dataset is not included. |
| Version | 0.1.0 |

## Intended use

- Research and education on Sindhi NLP, and as a starting point for sentiment tools on **short, single-sentence** Sindhi text.
- Input is expected to be Sindhi in Arabic script, cleaned with the project's `clean_text` (Unicode NFKC, invisible characters removed, whitespace normalized) before prediction.

## Out-of-scope uses

- Decisions about people (employment, credit, moderation or legal outcomes) without human review.
- Text far from the training data: long documents, social-media slang or code-mixed text, other languages or other Sindhi scripts, sarcasm-heavy text. These are untested.
- Measuring sentiment of a population or topic without checking accuracy on that kind of text first.

## Data

**Natural dataset (human-labeled).** 4,420 Sindhi sentences with three labels (1,500 negative, 1,419 neutral, 1,501 positive). Sentences are 2 to 54 words long (mean 13.3) with a vocabulary of 9,060 words. Split with a fixed seed, grouped so near-duplicates stay together: 3,094 train, 442 validation, 884 test (locked until the final evaluation).
Source and labeling: the sentences were collected from Awaz and other articles and newspapers, and labeled by a single human annotator. No second annotator reviewed the labels, so agreement between annotators was not measured. A formal definition of the neutral class is not documented here. The dataset is available separately on Kaggle: https://www.kaggle.com/datasets/alinawaz06/sindhi-sentiment-analysis-dataset (check the dataset page for its terms, since the text comes from published sources).

**Synthetic dataset.** 150,000 machine-generated sentences (50,000 per class), split 120,000 / 15,000 / 15,000 with duplicates and weekday variants of the same sentence kept within one split. The text is built from sentence templates with a small vocabulary (about 2,400 words in the training split). It is useful for testing the pipeline but **not representative of real Sindhi**: a baseline trained on it scores 0.9998 on its own validation set and only 0.6002 on the natural validation set.

## Training procedure (`xlmr_mixed`)

- Training data: up to 30,000 synthetic training sentences plus the natural training split repeated 5 times.
- Texts that also appear in a validation set were removed from training.
- Learning rate 2e-5, batch size 32, up to 3 epochs, warmup ratio 0.1, weight decay 0.01, max length 128, mixed precision (fp16).
- Checkpoint chosen by macro-F1 on the natural validation set (best epoch: 2).
- Hardware: one NVIDIA Tesla T4 on Kaggle.
- Training time and exact library versions: see `artifacts/models/xlmr_mixed/training_config.json`.

The baseline was trained on the natural training split only, in under one second on CPU.

## Evaluation

Scored on the natural dataset. Test results were produced once, after all choices were fixed.

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
| --- | ---: | ---: | ---: |
| TF-IDF + LinearSVC, trained on synthetic data | 0.6002 | not run | not run |
| TF-IDF + LinearSVC, trained on natural data | 0.8665 | 0.8695 | 0.8688 |
| XLM-RoBERTa, natural data only | 0.8663 | not run | not run |
| **XLM-RoBERTa, synthetic + natural (`xlmr_mixed`)** | **0.8872** | **0.8906** | **0.8903** |

Notes:

- The test set has 884 sentences, so each accuracy carries a margin of roughly ±2 percentage points. The 0.021 macro-F1 gap between the two main models should be read with that in mind. A paired comparison (exact McNemar test, saved in `artifacts/metrics/compare_baseline_natural_vs_xlmr_mixed_external_test.json`) found that on the 884 test sentences both models were right on 712 and wrong on 41. The baseline alone was wrong on 75 and XLM-R alone on 56, giving **p = 0.116**. The difference between the two models is therefore **not statistically significant** at this test size: XLM-R scored higher, but the result could plausibly be chance. When compute is limited, the lightweight baseline is a reasonable choice.
- Per-class precision, recall, F1 and confusion matrices are in `artifacts/metrics/`.
- Scores on the synthetic validation set (0.9998 for every model) are not evidence of quality.

## Limitations and risks

- **Small, narrow evaluation.** One human-labeled set of short sentences. Performance on other domains, longer text, or other authors is unknown.
- **Label quality is unmeasured.** The labels come from one annotator and were not independently reviewed, so inter-annotator agreement is unknown. Neutral is often the least well-defined class.
- **Synthetic data in training.** The mixed model saw templated sentences. The validation gain from adding them (0.8872 vs 0.8663 without) is small and not shown to be significant.
- **Topic cues.** The baseline's strongest neutral features are topic words (city, market, hospital, budget), which is consistent with news-style sources. Neutral predictions may follow topic instead of sentiment on unfamiliar subjects.
- **Overfitting signs.** In the mixed run, training loss fell to near zero while validation loss rose after epoch 2.
- **Language variety.** Behaviour on dialects, spelling variants, and transliterated Sindhi has not been tested.

## License

The trained model weights in this repository are free to use, including for commercial work, under the MIT License. Please keep the author credit (Ali Nawaz). The base model, `xlm-roberta-base`, is also MIT licensed. The 4,420-sentence human-labeled dataset is **not** part of this model release and is not covered by this license; it is published separately on Kaggle (link above) under its own terms.

## Ethical considerations

Sentiment labels are a coarse and subjective summary of text. Predictions can be wrong, and errors may not be evenly distributed across topics, dialects, or communities. Use human review for anything consequential. The natural dataset contains sentences taken from newspapers and articles, so check its terms before reusing or redistributing it.

## How to use (until the SDK is released)

```python
from transformers import pipeline
from sindhi_nlp.data.clean_data import clean_text

classifier = pipeline("text-classification", model="artifacts/models/xlmr_mixed")
print(classifier(clean_text("اڄ جو ڏينهن تمام سٺو آهي")))
```

## Files

Model folder: `artifacts/models/xlmr_mixed/` (weights, tokenizer, `label_mapping.json`, `training_config.json`, `metrics.json`, `training_log.jsonl`).
Reports: `artifacts/metrics/`, `artifacts/reports/`.