# SUBSET OF THE WIKIPEDIA SUMMARY DATASET.

This  repo is used to sample a subset of summaries from wikipedia. It is useful for research into machine learning and natural language processing. It contains all titles and summaries (or introductions) of English Wikipedia articles, extracted from the repository [Wikipedia-Summary-Dataset](https://github.com/tscheepers/Wikipedia-Summary-Dataset). 

We use the tokenized subset from here: [💾 **tokenized.tar.gz**](http://blob.thijs.ai/wiki-summary-dataset/tokenized.tar.gz) (± 1GB; 533,211,092 words; 5,627,475 vocab; 5,315,384 articles).

Further, we use a subset of size k, (k=1000 for preliminary tasks) sampled from the above dataset.

Example Usage
-----

Example from `tokenized.txt`:

```
Anarchism ||| Anarchism is a political philosophy that advocates self-governed societies based on voluntary…
Autism ||| Autism is a neurodevelopmental disorder characterized by impaired social interaction , impaired verbal…
Albedo ||| Albedo ( ) is a measure for reflectance or optical brightness ( Latin albedo , `` whiteness '' ) of…
…
```

Dataset construction
-----

The dataset was constructed using a script that calls Wikipedia API for every page with their `page_id`.