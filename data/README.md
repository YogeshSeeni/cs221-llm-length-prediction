# data/

Contents of this directory are gitignored. The scripts create:

```
data/
  dataset.parquet       50k Vicuna-13B English single-turn conversations with token counts, bucket, and split (src/prepare_data.py)
  test_scores.parquet   test split with each predictor's ranking score (src/predict.py)
```

LMSYS-Chat-1M is gated on Hugging Face: accept the dataset terms, then run `huggingface-cli login`. The raw download goes to the Hugging Face cache, not this directory.
