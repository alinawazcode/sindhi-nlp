# Tutorials

Step-by-step walk-throughs, each ending with something you can check.

| Tutorial | You will | Needs |
| --- | --- | --- |
| [Classify a CSV file](classify-csv.md) | score every row of a spreadsheet and review the uncertain ones | `sindhi-nlp[transformers]` |
| [Evaluate on your own data](evaluate-own-data.md) | measure a model on sentences you labeled, find its worst mistakes | a labeled CSV |
| [Train on your own data](train-own-data.md) | prepare splits, train the baseline in seconds, optionally fine-tune XLM-R | a labeled CSV; a GPU for XLM-R |
| [Serve the API](serve-api.md) | run the HTTP service and call it from Python and curl | `sindhi-nlp[api,transformers]` |

Conventions: commands are shown for a Unix shell; on Windows PowerShell they are the same unless noted. Output blocks marked *toy model* come from real runs of a tiny model trained for these docs, to show the exact format. Numbers from the published model are only quoted from the [Results](../results.md) page.
