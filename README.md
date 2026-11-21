# Learned Output-Length Prediction for LLM Serving

Stanford CS 221 final project. Author: Yogesh Seenichamy.

## Problem

An LLM server does not know how long a response will be until generation finishes. Under first-come-first-served (FCFS), one 1,500-token essay can hold up many short answers. Shortest-job-first (SJF) minimizes mean latency when job sizes are known, but they are not.

This project predicts, from the prompt alone, a response-length bucket (`<64`, `64–256`, `256–1k`, `>1k` tokens) and a ranking score, then uses those predictions to order requests in a serving queue. The target model is Vicuna-13B, using its subset of LMSYS-Chat-1M. The question is how much ordering information a cheap, prompt-only predictor can recover and how much of the FCFS-to-oracle latency gap it closes.

See `docs/proposal.pdf` for the full proposal.

## Data

- LMSYS-Chat-1M, Vicuna-13B conversations (about 491k).
- 50k English, single-turn conversations, split 40k / 5k / 5k into train / validation / test by conversation ID.
- Labels: response length in tokens (Vicuna-13B v1.5 tokenizer), bucketed.

## Methods

**Baselines and oracle**
- FCFS (the vLLM / Orca default).
- Shortest-prompt-first (prompt length as a proxy for response length).
- Oracle SJF using true response lengths (upper bound).

**Predictors**
1. Logistic regression, then gradient boosting, on handcrafted features: prompt length, leading verb (write/list vs. is/what), explicit length requests ("500 words"), code markers, TF-IDF over the first 50 tokens. The expected bucket index is the ranking score.
2. Fine-tuned DistilBERT with an ordinal or pairwise RankNet loss.

**Scheduling**
- Predicted-SJF as the starting order.
- Local search over adjacent swaps minimizing a cost with a starvation penalty:
  `J(π) = Σ_i Ĉ_i + λ · Σ_i max(0, Ŵ_i − W_max)²`
- Sweep λ to trace the mean-latency vs. p99-latency tradeoff.
- Service-time model: `a · prompt_tokens + b · output_tokens`. Ties broken by ranking score, then arrival time.

## Evaluation

**Prediction quality**
- Four-class accuracy.
- Macro-F1 (the `>1k` bucket is rare).
- Kendall's tau between predicted scores and true lengths.

**Scheduling quality** (simulator, Poisson arrivals, one non-preemptive server)
- Mean latency and p99 latency.
- Fraction of the FCFS-to-oracle gap closed at light, moderate, and near-saturation loads.
- 5k test prompts replayed with five arrival seeds per load.

## Repository layout

```
docs/                  proposal and progress report (PDF and LaTeX source)
data/                  datasets (gitignored, see data/README.md)
src/prepare_data.py    build the 50k subset, tokenize, bucket, split; writes dataset stats
src/predict.py         features, logistic regression and gradient boosting, prediction metrics
src/simulator.py       single-server non-preemptive queue simulator and scheduling policies
src/run_all.py         replay test prompts under every policy; writes scheduling metrics and figures
results/               metrics (JSON) and figures
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LMSYS-Chat-1M is gated on Hugging Face. Accept the dataset terms, then run `huggingface-cli login` before downloading.

## Reproducing the results

Run from the repository root, in this order:

```bash
python src/prepare_data.py   # downloads LMSYS-Chat-1M; writes data/dataset.parquet and results/dataset_stats.json
python src/predict.py        # trains the predictors; writes results/prediction_metrics.json and data/test_scores.parquet
python src/run_all.py        # runs the simulator; writes results/scheduling_metrics.json and results/figures/
python src/simulator.py      # optional self-check against the worked example in the proposal
```

## Status

Milestones:

1. Done: access the dataset, filter to Vicuna-13B English single-turn, tokenize responses, build splits.
2. Done: simulator with FCFS, shortest-prompt-first, and oracle SJF.
3. Done: feature-based predictors (logistic regression, gradient boosting).
4. Not started: DistilBERT ranker.
5. Not started: predicted-SJF plus starvation-aware local search; λ sweep.

Results so far are in `docs/progress_report.pdf`. On the 5k test set the gradient-boosted model reaches Kendall's tau of 0.62 and closes 73–80% of the FCFS-to-oracle gap in mean latency, while p99 latency at load 0.95 rises from 265 s (FCFS) to 912 s.

## Known challenges

- Label noise: the same prompt resamples to different lengths, which caps achievable tau.
- Capped tail: 4% of responses are exactly 512 tokens (a generation cap in the deployed service) and only 0.24% exceed 1k tokens, so this data lacks the heavy tail that makes SJF most valuable.
- Starvation: predicted SJF makes p99 latency much worse than FCFS at high load.
