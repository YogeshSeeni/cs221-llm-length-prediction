"""Replay the test prompts through the simulator under each policy; write metrics and figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator import run_policies

LOADS, SEEDS = [0.5, 0.8, 0.95], range(5)
LABELS = {"fcfs": "FCFS", "shortest_prompt": "Shortest prompt", "pred_logreg": "Predicted SJF (LR)",
          "pred_gbm": "Predicted SJF (GBM)", "oracle_bucket": "Oracle bucket SJF", "oracle_sjf": "Oracle SJF"}

te = pd.read_parquet("data/test_scores.parquet")
scores = {"pred_logreg": te.score_logreg.values, "pred_gbm": te.score_gbm.values, "oracle_bucket": te.bucket.values}

rows = []
for load in LOADS:
    for seed in SEEDS:
        lat = run_policies(te.prompt_tokens.values, te.output_tokens.values, scores, load, seed)
        long_jobs = te.output_tokens.values >= np.quantile(te.output_tokens, 0.9)
        for name, l in lat.items():
            rows.append({"load": load, "seed": seed, "policy": name, "mean": l.mean(), "p99": np.quantile(l, 0.99),
                         "mean_long_jobs": l[long_jobs].mean()})
runs = pd.DataFrame(rows)
wide = runs.pivot(index=["load", "seed"], columns="policy", values="mean")
gap = wide.rsub(wide.fcfs, axis=0).div(wide.fcfs - wide.oracle_sjf, axis=0)  # (FCFS - policy) / (FCFS - oracle)

summary = {}
for load in LOADS:
    r = runs[runs.load == load].groupby("policy")
    summary[str(load)] = {p: {"mean_latency": r["mean"].mean()[p], "mean_latency_std": r["mean"].std()[p],
                              "p99_latency": r["p99"].mean()[p], "p99_latency_std": r["p99"].std()[p],
                              "mean_latency_long_jobs": r["mean_long_jobs"].mean()[p],
                              "gap_closed": gap.loc[load][p].mean(), "gap_closed_std": gap.loc[load][p].std()}
                          for p in LABELS}
json.dump(summary, open("results/scheduling_metrics.json", "w"), indent=2, default=float)
for load, d in summary.items():
    print(f"load {load}")
    for p, m in d.items():
        print(f"  {LABELS[p]:22s} mean {m['mean_latency']:8.1f}  p99 {m['p99_latency']:8.1f}  long {m['mean_latency_long_jobs']:8.1f}  gap {m['gap_closed']:.3f}")

# Figures
df = pd.read_parquet("data/dataset.parquet")
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.hist(df.output_tokens, bins=np.arange(0, 1150, 16), color="#4477aa")
for e in (64, 256, 1000):
    ax.axvline(e, color="k", ls="--", lw=0.8)
ax.set(xlabel="Response length (tokens)", ylabel="Conversations")
fig.savefig("results/figures/length_hist.pdf", bbox_inches="tight")

cm = np.array(json.load(open("results/prediction_metrics.json"))["gbm"]["confusion"], float)
fig, ax = plt.subplots(figsize=(2.6, 2.2))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
names = ["<64", "64-256", "256-1k", ">1k"]
ax.set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True")
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="w" if cm[i, j] / cm[i].sum() > 0.5 else "k")
fig.savefig("results/figures/confusion_gbm.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3.4, 2.2))
for p, c in [("shortest_prompt", "#bbbbbb"), ("pred_logreg", "#66ccee"), ("pred_gbm", "#4477aa"), ("oracle_bucket", "#ee6677")]:
    g = gap[p].groupby("load")
    ax.errorbar(LOADS, g.mean(), yerr=g.std(), marker="o", ms=3, capsize=2, color=c, label=LABELS[p])
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Load $\\rho$", ylabel="Gap closed", xticks=LOADS)
ax.legend(fontsize=6.5, frameon=False)
fig.savefig("results/figures/gap_closed.pdf", bbox_inches="tight")
