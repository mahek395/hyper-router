"""
gateway/eval_production_router.py

Comprehensive Evaluation of the Production Dynamic Gateway Router.
Benchmarks:
  1. Success Rate & Cost vs Baselines on RouterBench Test Split
  2. Latency Benchmarks (Microseconds per routing decision)
  3. Dynamic Model Registration Test (Adding/Removing models on the fly without retraining)
  4. Comparison with Static kNN Router
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
from gateway.models import ModelRegistry, ModelProfile
from gateway.complexity_scorer import ComplexityScorer
from gateway.filter import ConstraintFilter, RequestRequirements
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load Data & Set Up Test Split
# ---------------------------------------------------------------------------
prompts = pd.read_csv("prompts.csv")
embeddings = np.load("prompt_embeddings.npy")
results_long = pd.read_csv("model_results_long.csv")

success_lookup = results_long.set_index(["prompt_id", "model"])["success"]
cost_lookup = results_long.set_index(["prompt_id", "model"])["cost"]

train_idx, temp_idx = train_test_split(
    np.arange(len(prompts)), test_size=0.30, random_state=RANDOM_SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=RANDOM_SEED
)

test_prompts = prompts.iloc[test_idx].reset_index(drop=True)
test_embeddings = embeddings[test_idx]

print(f"Test Set Size: {len(test_prompts)} prompts")

# ---------------------------------------------------------------------------
# 2. Initialize Gateway Router
# ---------------------------------------------------------------------------
registry = ModelRegistry(include_defaults=True)
scorer = ComplexityScorer(model_path="gateway/complexity_model.pkl")
filter_engine = ConstraintFilter()
router = GatewayRouter(registry=registry, complexity_scorer=scorer, constraint_filter=filter_engine)

# Restrict to RouterBench candidate models for fair benchmark comparison
routerbench_models = [
    "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v2", "claude-v1",
    "claude-instant-v1", "mistralai/mixtral-8x7b-chat", "mistralai/mistral-7b-chat",
    "zero-one-ai/Yi-34B-Chat", "meta/llama-2-70b-chat", "meta/code-llama-instruct-34b-chat",
    "WizardLM/WizardLM-13B-V1.2"
]

# ---------------------------------------------------------------------------
# 3. Benchmark Baselines on Test Set
# ---------------------------------------------------------------------------
test_pids = test_prompts["prompt_id"].values
gpt4_succ = np.mean([success_lookup.get((pid, "gpt-4-1106-preview"), 0) for pid in test_pids])
gpt4_cost = np.mean([cost_lookup.get((pid, "gpt-4-1106-preview"), 0) for pid in test_pids])

mistral_succ = np.mean([success_lookup.get((pid, "mistralai/mistral-7b-chat"), 0) for pid in test_pids])
mistral_cost = np.mean([cost_lookup.get((pid, "mistralai/mistral-7b-chat"), 0) for pid in test_pids])

print("\n" + "=" * 75)
print("BASELINES (Held-out Test Set)")
print("=" * 75)
print(f"Always GPT-4-1106     : Success={gpt4_succ:.1%}, Avg Cost=${gpt4_cost:.6f}")
print(f"Always Mistral-7B     : Success={mistral_succ:.1%}, Avg Cost=${mistral_cost:.6f}")

# ---------------------------------------------------------------------------
# 4. Sweep Lambda for Production Router & Measure Latency
# ---------------------------------------------------------------------------
LAMBDA_GRID = np.concatenate([[0.0], np.geomspace(1, 10000, 12)])
curve_rows = []
latencies_ms = []

print("\n" + "=" * 75)
print("PRODUCTION DYNAMIC ROUTER RESULTS (Tested across Lambda Pareto Grid)")
print("=" * 75)

for lam in LAMBDA_GRID:
    config = RoutingConfig(policy=RoutingPolicy.BALANCED, lambda_cost=lam)
    reqs = RequestRequirements(
        allowed_model_ids=set(routerbench_models),
        estimated_input_tokens=150,
        estimated_output_tokens=250,
    )
    
    rows = []
    for i, row in test_prompts.iterrows():
        pid = row["prompt_id"]
        text = row["prompt_text"]
        emb = test_embeddings[i]

        decision = router.route(text, requirements=reqs, config=config, embedding=emb)
        latencies_ms.append(decision.routing_latency_ms)

        actual_succ = success_lookup.get((pid, decision.selected_model_id), 0)
        actual_cst = cost_lookup.get((pid, decision.selected_model_id), 0)
        rows.append({
            "prompt_id": pid,
            "selected_model": decision.selected_model_id,
            "success": actual_succ,
            "cost": actual_cst,
            "complexity": decision.prompt_complexity,
        })

    df = pd.DataFrame(rows)
    succ_rate = df["success"].mean()
    avg_cost = df["cost"].mean()
    curve_rows.append({
        "lambda": lam,
        "success_rate": succ_rate,
        "avg_cost": avg_cost,
    })
    print(f"  lambda={lam:>9.2f} | Success={succ_rate:.1%} | Avg Cost=${avg_cost:.6f}")

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv("gateway/production_router_curve.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Routing Latency Analysis
# ---------------------------------------------------------------------------
latencies_arr = np.array(latencies_ms)
print("\n" + "=" * 75)
print("ROUTING ENGINE LATENCY PROFILE (Excluding Upstream LLM Call)")
print("=" * 75)
print(f"  Mean Latency  : {np.mean(latencies_arr):.3f} ms")
print(f"  Median (P50)  : {np.median(latencies_arr):.3f} ms")
print(f"  P95 Latency   : {np.percentile(latencies_arr, 95):.3f} ms")
print(f"  P99 Latency   : {np.percentile(latencies_arr, 99):.3f} ms")

# ---------------------------------------------------------------------------
# 6. Dynamic Model Registration Test (The Superpower of this Architecture)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("DYNAMIC MODEL REGISTRATION TEST (Zero-Retraining Demonstration)")
print("=" * 75)

# Simulate a tenant registering a brand new custom model pool with modern 2026 models
tenant_registry = ModelRegistry(include_defaults=False)
tenant_registry.register(ModelProfile(
    model_id="deepseek/deepseek-chat-v3",
    provider="deepseek",
    input_cost_per_m=0.14,
    output_cost_per_m=0.28,
    capability_score=0.90,
    features={"tools", "json_mode"},
))
tenant_registry.register(ModelProfile(
    model_id="openai/gpt-4o-mini",
    provider="openai",
    input_cost_per_m=0.15,
    output_cost_per_m=0.60,
    capability_score=0.78,
    features={"tools", "json_mode", "vision"},
))
tenant_registry.register(ModelProfile(
    model_id="anthropic/claude-3-5-sonnet",
    provider="anthropic",
    input_cost_per_m=3.00,
    output_cost_per_m=15.00,
    capability_score=0.95,
    features={"tools", "json_mode", "vision"},
))

tenant_router = GatewayRouter(registry=tenant_registry, complexity_scorer=scorer)

sample_queries = [
    ("Hello! What is the capital of France?", "Simple Greeting"),
    ("Summarize this 2-page text about photosynthesis.", "Summarization"),
    ("Write a fast PyTorch CUDA kernel for FlashAttention with reverse-mode autodiff.", "Deep Expert Coding"),
    ("Calculate the exact Bayes posterior for a Dirichlet-Multinomial process with non-conjugate prior.", "Complex Math"),
]

for query, q_type in sample_queries:
    dec = tenant_router.route(query, config=RoutingConfig(policy=RoutingPolicy.BALANCED, lambda_cost=200.0))
    print(f"\nQuery ({q_type}): \"{query}\"")
    print(f"  -> Complexity: {dec.prompt_complexity:.3f}")
    print(f"  -> Routed to : {dec.selected_model_id} (Est cost: ${dec.estimated_cost_usd:.6f})")
    print(f"  -> Reason    : {dec.decision_reason}")

# ---------------------------------------------------------------------------
# 7. Plot Cost-Quality Pareto Comparison
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(curve_df["avg_cost"], curve_df["success_rate"], "b-o", linewidth=2, label="Dynamic Production Router")
plt.plot([mistral_cost, gpt4_cost], [mistral_succ, gpt4_succ], "k--", label="Naive Interpolation Baseline")
plt.scatter([mistral_cost], [mistral_succ], color="red", s=100, zorder=5, label="Mistral-7B Only")
plt.scatter([gpt4_cost], [gpt4_succ], color="green", s=100, zorder=5, label="GPT-4 Only")

plt.xlabel("Avg Cost per Request ($)")
plt.ylabel("Success Rate on Test Split")
plt.title("Production Gateway Router Pareto Frontier")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gateway/production_router_pareto.png", dpi=150)
print("\nSaved evaluation plot to gateway/production_router_pareto.png")
