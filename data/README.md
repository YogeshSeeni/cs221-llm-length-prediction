# data/

Contents of this directory are gitignored. Expected layout:

```
data/
  raw/         LMSYS-Chat-1M download (gated on Hugging Face; accept terms, then `huggingface-cli login`)
  processed/   Vicuna-13B English single-turn subset, tokenized, with length buckets
  splits/      train/val/test (40k/5k/5k) split by conversation ID
```
