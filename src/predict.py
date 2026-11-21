"""Prompt-only response-length predictors: features, models, and prediction metrics."""
import json
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

LONG_VERBS = {"write", "explain", "describe", "generate", "create", "list", "give", "tell", "compose", "summarize"}
SHORT_STARTS = {"is", "are", "what", "who", "when", "where", "which", "does", "do", "can", "hi", "hello", "say"}
HAND_NAMES = ["log_prompt_tokens", "log_chars", "n_lines", "n_questions", "long_verb", "short_start",
              "has_number_request", "log_requested_words", "asks_brief", "asks_detailed", "has_code",
              "wants_essay", "wants_list", "one_word_answer"]


def first_words(prompt, n=50):
    return " ".join(prompt.split()[:n]).lower()


def hand_features(df):
    rows = []
    for prompt, n_tok in zip(df.prompt, df.prompt_tokens):
        low = prompt.lower()
        first = re.findall(r"[a-z']+", low[:40])[:1]
        req = re.search(r"(\d[\d,]*)\s*[- ]?(words?|sentences?|paragraphs?|lines?|characters?)", low)
        words = 0.0
        if req:
            scale = {"w": 1, "s": 20, "p": 100, "l": 12, "c": 0.2}[req.group(2)[0]]
            words = min(float(req.group(1).replace(",", "")), 1e5) * scale
        rows.append([
            np.log1p(n_tok), np.log1p(len(prompt)), prompt.count("\n"), prompt.count("?"),
            bool(first) and first[0] in LONG_VERBS, bool(first) and first[0] in SHORT_STARTS,
            req is not None, np.log1p(words),
            bool(re.search(r"\b(brief|briefly|short|concise|one sentence|in a sentence)\b", low)),
            bool(re.search(r"\b(detailed|in detail|comprehensive|elaborate|step by step|long)\b", low)),
            "```" in prompt or bool(re.search(r"\b(def |class |function|import |#include|select .* from)\b", low)),
            bool(re.search(r"\b(essay|article|story|poem|blog|report|introduction|letter|script)\b", low)),
            bool(re.search(r"\b(list|steps|examples|ideas|top \d+)\b", low)),
            bool(re.search(r"\b(yes or no|one word|true or false|only answer|single word|just the)\b", low)),
        ])
    return np.array(rows, float)


def metrics(y, lengths, proba):
    score = proba @ np.arange(proba.shape[1])  # expected bucket index
    pred = proba.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
            "kendall_tau": kendalltau(score, lengths).statistic}, score


if __name__ == "__main__":
    df = pd.read_parquet("data/dataset.parquet")
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    te["seen_in_train"] = te.prompt.isin(set(tr.prompt))

    scaler = StandardScaler().fit(hand_features(tr))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20_000, sublinear_tf=True)
    tfidf.fit(tr.prompt.map(first_words))
    X = lambda d: hstack([csr_matrix(scaler.transform(hand_features(d))), tfidf.transform(d.prompt.map(first_words))]).tocsr()
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    length_only = lambda d: scaler.transform(hand_features(d))[:, :1]

    def onehot(n, k):
        p = np.zeros((n, 4))
        p[:, k] = 1
        return p

    # Pick each model's regularisation on validation Kendall tau.
    def best(candidates, fit):
        fitted = [fit(c) for c in candidates]
        taus = [metrics(va.bucket, va.output_tokens, m.predict_proba(Xva))[0]["kendall_tau"] for m in fitted]
        i = int(np.argmax(taus))
        return fitted[i], candidates[i]

    lr, lr_c = best([0.1, 1.0, 10.0], lambda c: LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(Xtr, tr.bucket))
    gbm, gbm_leaves = best([15, 31, 63], lambda n: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=n, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, class_weight="balanced", random_state=0, verbose=-1).fit(Xtr, tr.bucket))
    len_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(length_only(tr), tr.bucket)

    probas = {
        "majority": onehot(len(te), tr.bucket.mode()[0]),
        "prompt_length_lr": len_lr.predict_proba(length_only(te)),
        "logreg": lr.predict_proba(Xte),
        "gbm": gbm.predict_proba(Xte),
    }
    unseen = ~te.seen_in_train.values
    results, scores = {"chosen": {"logreg_C": lr_c, "gbm_num_leaves": gbm_leaves},
                       "test_seen_in_train_fraction": float(te.seen_in_train.mean())}, {}
    for name, p in probas.items():
        p4 = np.zeros((len(te), 4))
        p4[:, : p.shape[1]] = p
        results[name], scores[name] = metrics(te.bucket, te.output_tokens, p4)
        results[name]["unseen_prompts"] = metrics(te.bucket[unseen], te.output_tokens[unseen], p4[unseen])[0]
        results[name]["confusion"] = confusion_matrix(te.bucket, p4.argmax(1), labels=[0, 1, 2, 3]).tolist()
    results["majority"]["kendall_tau"] = results["majority"]["unseen_prompts"]["kendall_tau"] = 0.0
    results["raw_prompt_tokens_tau"] = kendalltau(te.prompt_tokens, te.output_tokens).statistic

    names = HAND_NAMES + list(tfidf.get_feature_names_out())
    top = np.argsort(gbm.booster_.feature_importance("gain"))[::-1][:15]
    results["gbm_top_features"] = [names[i] for i in top]

    json.dump(results, open("results/prediction_metrics.json", "w"), indent=2, default=float)
    out = te[["conversation_id", "prompt_tokens", "output_tokens", "bucket", "seen_in_train"]].copy()
    for name in ("logreg", "gbm"):
        out[f"score_{name}"] = scores[name]
    out.to_parquet("data/test_scores.parquet")
    for k, v in results.items():
        print(k, {a: b for a, b in v.items() if a != "confusion"} if isinstance(v, dict) else v)
