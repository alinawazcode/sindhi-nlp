# Datasets

This project uses two datasets with very different roles. Only the human-labeled one is used for reported results.

| | Human-labeled ("natural") | Synthetic |
| --- | --- | --- |
| Size | 4,420 sentences | 150,000 sentences |
| Labels | negative / neutral / positive | negative / neutral / positive |
| Class balance | 1,500 / 1,419 / 1,501 | 50,000 each |
| Origin | Awaz and other articles and newspapers | Generated from sentence templates |
| Labeled by | One human annotator (the author) | Assigned by the generator |
| Used for | Training, validation, test, all reported results | Pipeline testing; optional extra training data |
| In this repository | No (see below) | No |

## Human-labeled dataset

**Source.** Sindhi sentences collected from Awaz and other articles and newspapers, each labeled `negative`, `neutral`, or `positive`. Published on Kaggle: <https://www.kaggle.com/datasets/alinawaz06/sindhi-sentiment-analysis-dataset>.

**Schema.** A CSV with two columns, `Sindhi Text` and `Sentiment` (labels such as `Positive`). The pipeline reads them case-insensitively and maps them to lowercase labels. After preparation, each split has the columns `Text` and `Label`.

**Statistics.** Sentences are 2 to 54 words long (mean 13.3). The vocabulary is 9,060 distinct words. No empty texts, invalid labels, or duplicates were found.

**Splits** (stratified by label, seed 42, near-duplicates kept together):

| Split | Rows | Negative | Neutral | Positive | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 3,094 | 1,050 | 993 | 1,051 | Training |
| Validation | 442 | 150 | 142 | 150 | Choosing models and checkpoints |
| Test | 884 | 300 | 284 | 300 | Scored once per model |

**Quality and limits.**
- One annotator (the author) labeled the data. No second annotator reviewed it, so agreement between annotators is unknown.
- **Neutral class.** The annotator (the project author) labeled a sentence `neutral` when it reads as normal, everyday conversation rather than expressing a clearly positive or negative feeling. This is a working definition from one person, not a written annotation guideline. The baseline's strongest neutral features are topic words such as city, market, hospital, and budget, so a model may partly learn topic instead of sentiment for that class.
- **License.** The dataset is released on Kaggle under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): you may share and adapt it, including commercially, if you credit the author, link the license, mark changes, and release derived datasets under the same license. The sentences were collected from published sources, so consider how that affects your own use.
- None of the sentences also appear in the synthetic data's train or validation splits.

## Synthetic dataset

**What it is.** 150,000 machine-generated sentences, 50,000 per label. An audit of the data showed they are built from templates: each sentence combines a city, a place, an event, and a weekday, with an explicit sentiment phrase. Across the training split there are only about 2,400 distinct words, and many sentences are the same sentence with a different weekday.

**Splits.** Stratified 80/10/10 with seed 42: 120,000 train, 15,000 validation, 15,000 test, with equal classes in each. Texts that differ only by weekday are kept in the same split. Without this, 23.9% of validation sentences had a weekday twin in the training set.

**Why it is not used for conclusions.**
- Every model scores 0.9998 on its validation set, because each sentence carries an explicit sentiment phrase and the vocabulary is tiny.
- A TF-IDF model trained on it scores only **0.6002** macro-F1 on the human-labeled validation set, against **0.8665** when trained on the human-labeled training split. The synthetic data does not teach real Sindhi.
- The one model that uses it as extra training data (`xlmr_mixed`) improved over the natural-only XLM-R run by 0.021 validation macro-F1, a gap too small to call proven.

## Preparation

Both datasets go through the same steps, defined in `src/sindhi_nlp/data/`:

1. **Clean**: Unicode NFKC normalization, removal of zero-width and bidirectional control characters and tatweel, and whitespace normalization (`clean_data.py`). The same function runs at prediction time.
2. **Validate**: reject missing text and invalid labels, and count duplicates and conflicting labels (`validate_data.py`).
3. **Deduplicate**: drop texts that appear with conflicting labels, and keep the first copy of exact duplicates.
4. **Split**: stratified, seeded, grouped by near-duplicate key (`split_data.py`, `prepare_external.py`).
5. **Report**: each preparation run writes a `split_report.json` with row counts, label counts, source file hashes, and the cleaning settings used.

Commands to rebuild the splits are in the README.
