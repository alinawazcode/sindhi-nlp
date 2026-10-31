# Error analysis of the published model

What the 97 mistakes of `xlmr_mixed` on the 884 test sentences look like. This page reads the saved list of mistakes (`xlmr_mixed_external_test_misclassified.csv`), whose counts match the confusion matrix on the [Results](../results.md) page exactly. English glosses are rough machine translations (written by the AI assistant that drafted this page) to help readers who do not read Sindhi. They are not part of the data and should be checked by a Sindhi speaker.

Sentences are quoted from the *Sindhi Sentiment Analysis Dataset* by Ali Nawaz, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (<https://www.kaggle.com/datasets/alinawaz06/sindhi-sentiment-analysis-dataset>). The labels shown are the dataset's.

## The six kinds of mistake

| True label | Model said | Count |
| --- | --- | ---: |
| neutral | negative | 24 |
| positive | negative | 21 |
| negative | positive | 20 |
| neutral | positive | 15 |
| positive | neutral | 9 |
| negative | neutral | 8 |

Mistakes are not short-text artifacts. The mistaken sentences average 13.1 words against 13.3 for the whole dataset, so length does not explain them.

## Pattern 1: factual statements about a bad event (neutral called negative)

The largest group. The labeled class is `neutral` (an ordinary statement), but the content describes something unwelcome, and the model reads it as `negative`.

| Sentence | Rough meaning | Label | Model |
| --- | --- | --- | --- |
| بئنڪ اڄ ڪاروبار لاءِ نه کليل آهي. | The bank is not open for business today. | neutral | negative |
| ميٽنگ اڄ منسوخ ٿي وئي آهي | The meeting has been cancelled today. | neutral | negative |
| انٽرنيٽ جي رفتار اڄ سست آهي | The internet is slow today. | neutral | negative |
| سنڌوءَ جي پاڻيءَ جي سطح تمام گهٽجي وئي آهي. | The water level of the Indus has fallen a lot. | neutral | negative |

Whether such a sentence is neutral or negative depends on the labeling rule (here: normal, everyday conversation counts as neutral). A model trained on few thousand sentences can only approximate that boundary.

## Pattern 2: short, figurative expressions of sadness (negative called positive)

| Sentence | Rough meaning | Label | Model |
| --- | --- | --- | --- |
| دل اندر کان ڀاري آهي | The heart feels heavy inside. | negative | positive |
| ماءُ جي ياد ستائي ٿي | Missing my mother hurts. | negative | positive |
| پياري جي ياد ۾ دل روئندو آهي | The heart weeps remembering a loved one. | negative | positive |

These sentences contain no plainly negative word; the sadness is carried by an idiom. This is where a pretrained model's confident wrong answers are most surprising, and where the baseline, which counts words, tends to fail too.

## Pattern 3: plain achievements called negative (positive called negative)

| Sentence | Rough meaning | Label | Model |
| --- | --- | --- | --- |
| ننڍي ڀاءُ امتحان ۾ ٽاپ ڪيو | My younger brother topped the exam. | positive | negative |
| امتحان ۾ يونيورسٽي ٽاپ ڪيو | Topped the university in the exam. | positive | negative |

These look like genuine model errors, with no ambiguity in the label.

## Pattern 4: a cluster of product-launch sentences labeled negative

Four of the eight `negative` sentences the model called `neutral` are about a product being offered for sale in one place or on one date, or about no announcement for other countries. Two name a phone (for example: اها فون في الحال برازيل ۾ وڪري لاءِ پيش ڪئي وئي آهي ... , "This phone is currently on sale in Brazil ..."); the other two say only "it" or "this" and read as part of the same news item. They are factual statements. The dataset labels them `negative`, the model says `neutral`.

This is the kind of case worth **re-checking the labels** for: either the annotator saw a negative angle (unavailability elsewhere) that a reader cannot see from the sentence, or the labels are inconsistent with other factual sentences labeled `neutral`, such as those in Pattern 1.

## Compared with the baseline

The TF-IDF baseline made 116 mistakes on the same sentences (XLM-R: 97). Matching the two lists sentence by sentence gives the same agreement counts as the saved comparison: **41 sentences both models got wrong**, 56 only XLM-R got wrong, 75 only the baseline got wrong.

Of the 41 sentences both models got wrong, **33 get the same wrong label from both**. By true label: 18 neutral, 12 positive, 11 negative. The most common shared mistake is the Pattern 1 case: 11 neutral sentences that both models call negative. The next are 8 negative sentences both call positive and 6 positive sentences both call negative.

When two very different models, one reading whole sentences and one counting words, agree on the same wrong answer, a label problem is more likely than a model problem. These 41 sentences are the best place to start if you want to re-check labels. They are listed in a separate review file, `label_review_candidates.csv` (sentence, label, both models' predictions), which is not part of the repository.

## What this tells you

- **The boundaries between classes, especially neutral against negative, are where most mistakes are.** 56 of the 97 mistakes involve `neutral` as the true or the predicted label; the neutral/negative boundary alone accounts for 32.
- **Some mistakes may be label inconsistencies, not model errors.** With a single annotator and no written guideline, a second pass over the mistakes list is the cheapest way to raise the ceiling.
- **Mistakes are clustered.** Several come in near-paraphrase groups (three sentences about an open book, two about a closed bank, four about a product on sale). If the dataset contains many such groups, the test set holds fewer independent examples than 884, so the ±2 point uncertainty quoted elsewhere is probably optimistic.
- **Confidence cannot be analyzed from this file.** It stores only the text, true label and predicted label. The newer `evaluate` tool also saves the confidence of each mistake ([Evaluate on your own data](../tutorials/evaluate-own-data.md)), so a future analysis can check whether wrong answers are confident. The documented behavior of the model, 1.000 on both sentences in the [Quickstart](../quickstart.md), suggests they are.

## Do it yourself

```python
import pandas as pd

d = pd.read_csv("xlmr_mixed_external_test_misclassified.csv", encoding="utf-8-sig")
print(d.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False))
```
