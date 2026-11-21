"""Build the 50k Vicuna-13B English single-turn subset of LMSYS-Chat-1M."""
import glob
import json

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

SEED = 0
BUCKET_EDGES = [64, 256, 1000]
BUCKET_NAMES = ["<64", "64-256", "256-1k", ">1k"]

path = snapshot_download("lmsys/lmsys-chat-1m", repo_type="dataset", allow_patterns="data/*.parquet")
cols = ["conversation_id", "model", "conversation", "turn", "language"]
df = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{path}/data/*.parquet")))

funnel = {"all": len(df)}
df = df[df.model == "vicuna-13b"]
funnel["vicuna-13b"] = len(df)
df = df[df.language == "English"]
funnel["english"] = len(df)
df = df[df.turn == 1]
funnel["single_turn"] = len(df)
df["prompt"] = df.conversation.map(lambda c: c[0]["content"])
df["response"] = df.conversation.map(lambda c: c[1]["content"])
df = df[(df.prompt.str.strip() != "") & (df.response.str.strip() != "")]
funnel["non_empty"] = len(df)

df = df.sort_values("conversation_id").sample(50_000, random_state=SEED).reset_index(drop=True)
tok = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
count = lambda texts: [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]
df["prompt_tokens"] = count(df.prompt)
df["output_tokens"] = count(df.response)
df["bucket"] = np.digitize(df.output_tokens, BUCKET_EDGES)
df["split"] = ["train"] * 40_000 + ["val"] * 5_000 + ["test"] * 5_000
assert df.conversation_id.is_unique

keep = ["conversation_id", "split", "prompt", "prompt_tokens", "output_tokens", "bucket"]
df[keep].to_parquet("data/dataset.parquet")

out = df.output_tokens
stats = {
    "funnel": funnel,
    "splits": df.split.value_counts().to_dict(),
    "bucket_fraction": {
        s: dict(zip(BUCKET_NAMES, np.bincount(g.bucket, minlength=4) / len(g))) for s, g in df.groupby("split")
    },
    "output_tokens": {"mean": out.mean(), "median": out.median(), "p90": out.quantile(0.9),
                      "p99": out.quantile(0.99), "max": int(out.max())},
    "prompt_tokens": {"mean": df.prompt_tokens.mean(), "median": df.prompt_tokens.median(),
                      "p99": df.prompt_tokens.quantile(0.99), "max": int(df.prompt_tokens.max())},
    "duplicate_prompt_fraction": float(df.prompt.duplicated(keep=False).mean()),
    "most_common_output_lengths": out.value_counts().head(5).to_dict(),
    "share_of_output_tokens_from_top_1pct": float(out.nlargest(len(out) // 100).sum() / out.sum()),
}
json.dump(stats, open("results/dataset_stats.json", "w"), indent=2, default=float)
print(json.dumps(stats, indent=2, default=float))
