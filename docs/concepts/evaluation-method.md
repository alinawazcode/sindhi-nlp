# Evaluation method

How the numbers in this project were produced, and why they can be trusted more than a single headline score.

## Two datasets, two roles

A 150,000-sentence **synthetic** dataset was built from sentence templates. Every model scored 0.9998 on it, including a baseline that then scored 0.60 on human-written sentences. The synthetic score measures how well a model memorizes templates, not whether it understands Sindhi. So:

- all reported results use a separate **human-labeled** set of 4,420 sentences;
- synthetic data is used only to test the pipeline and, in one model, as extra training data.

See [Dataset](../dataset.md).

## Splits that cannot leak

| Rule | Why |
| --- | --- |
| Stratified by label, fixed seed (42) | each split has the same class mix; reruns give identical files |
| Texts that are identical after normalization stay in one split | a sentence cannot be in both train and test |
| Near-duplicates are grouped (configurable `data.grouping` rules, e.g. the same sentence with a different weekday) | template copies cannot leak |
| Duplicates with conflicting labels are dropped | they cannot both be right |
| Texts in a validation set are removed from training data | no contamination |
| `split_report.json` stores counts, seed, file hashes and cleaning options | results can be tied to exact input files |

## The test split is used once

Choices (model type, epochs, which checkpoint, thresholds) are made on the **validation** split. The test split is scored once per model after the choices are fixed. The tooling enforces it: `evaluate_external --split test` needs `--final`, and a second run for the same model is refused unless `--overwrite`. Training code can read only `train.csv` and `validation.csv`.

This matters because every look at the test set while still deciding turns it into a second validation set, and the final number becomes optimistic.

## Metrics

- **Macro-F1** is the headline metric: the F1 of each class averaged with equal weight, so a weak class cannot hide behind strong ones.
- **Accuracy** is reported next to it.
- **Per-class precision, recall, F1 and the confusion matrix** show where errors happen.

## Uncertainty

With 884 test sentences, accuracy near 0.89 has a 95% **Wilson interval** of about ±2 points (baseline 0.845 to 0.889, XLM-R 0.868 to 0.909). The intervals overlap, so single numbers should not be read as exact.

## Comparing two models: McNemar's exact test

Both models are scored on the same sentences, so the right comparison is **paired**: look only at sentences where they disagree. Let *b* be sentences only model A got wrong and *c* sentences only model B got wrong. Under "no real difference", each disagreement is equally likely to go either way, so *b* follows a Binomial(*b* + *c*, 0.5) distribution. The exact two-sided p-value is the probability of a split at least as lopsided.

For the published models: both right on 712, both wrong on 41, only the baseline wrong on 75, only XLM-R wrong on 56. With 131 disagreements the exact p-value is **0.1155**, above the usual 0.05, so the data do **not** show XLM-R is better. It scored higher; chance is a plausible explanation.

The comparison tool uses only saved error lists, so it loads no models and cannot trigger a second test-set evaluation.

## What this evaluation does not tell you

- It is one dataset from one source, labeled by one person; inter-annotator agreement is unknown.
- Out-of-domain performance (social media, long text, other scripts) is untested.
- The estimate is about this test set. Measure on your own data: [Evaluate on your own data](../tutorials/evaluate-own-data.md).
