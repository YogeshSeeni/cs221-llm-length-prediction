"""Single-server, non-preemptive queue simulator for LLM request scheduling."""
import heapq

import numpy as np

# Service-time model in seconds: A per prompt token (prefill), B per output token (decode).
A, B = 0.0005, 0.02


def service_time(prompt_tokens, output_tokens):
    return A * np.asarray(prompt_tokens, float) + B * np.asarray(output_tokens, float)


def poisson_arrivals(n, rate, seed):
    return np.cumsum(np.random.default_rng(seed).exponential(1.0 / rate, n))


def simulate(arrival, service, priority):
    """Serve jobs one at a time; when the server frees up, pick the waiting job
    with the smallest priority (ties: earlier arrival). Returns per-job latency
    (completion time minus arrival time)."""
    n = len(arrival)
    order = np.argsort(arrival, kind="stable")
    latency = np.zeros(n)
    queue, t, nxt = [], 0.0, 0
    for _ in range(n):
        if not queue and arrival[order[nxt]] > t:
            t = arrival[order[nxt]]
        while nxt < n and arrival[order[nxt]] <= t:
            i = order[nxt]
            heapq.heappush(queue, (priority[i], arrival[i], i))
            nxt += 1
        _, _, i = heapq.heappop(queue)
        t += service[i]
        latency[i] = t - arrival[i]
    return latency


def run_policies(prompt_tokens, output_tokens, scores, load, seed):
    """Latencies for every policy at one load and one arrival seed.
    `scores` maps a predictor name to its ranking score per job."""
    service = service_time(prompt_tokens, output_tokens)
    arrival = poisson_arrivals(len(service), load / service.mean(), seed)
    policies = {"fcfs": arrival, "shortest_prompt": prompt_tokens, "oracle_sjf": service, **scores}
    return {name: simulate(arrival, service, np.asarray(p, float)) for name, p in policies.items()}


if __name__ == "__main__":
    # Worked example from the proposal: four jobs arrive together, one time unit per output token.
    out = np.array([20.0, 800.0, 50.0, 300.0])
    arrival = np.zeros(4)
    bucket = np.digitize(out, [64, 256, 1000])
    fcfs = simulate(arrival + np.arange(4) * 1e-9, out, np.arange(4.0)).mean()
    oracle = simulate(arrival, out, out).mean()
    # Bucketed SJF: 300 and 800 share a bucket, so the tie is broken the wrong way here.
    bucketed = simulate(arrival, out, bucket + np.array([0, 0, 0, 0.5])).mean()
    print(f"FCFS {fcfs:.1f}  oracle SJF {oracle:.1f}  bucketed SJF {bucketed:.1f}")
    assert np.allclose([fcfs, oracle, bucketed], [720.0, 407.5, 532.5])  # 720 / 408 / 533 in the proposal
    print("ok")
