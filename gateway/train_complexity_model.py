"""
gateway/train_complexity_model.py

Trains a decoupled, ultra-fast Prompt Complexity Regressor using RouterBench failure curves.
Maps any incoming prompt embedding/text to an intrinsic difficulty score C in [0.0, 1.0].
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
prompts = pd.read_csv("prompts.csv")
embeddings = np.load("prompt_embeddings.npy")
results_long = pd.read_csv("model_results_long.csv")

print(f"Loaded {len(prompts)} prompts and {len(results_long)} model evaluation rows.")

# ---------------------------------------------------------------------------
# 2. Compute Ground-Truth Prompt Difficulty / Complexity Metric
# ---------------------------------------------------------------------------
# Model capability calibration weights (based on overall success rate on benchmark)
model_success_rates = results_long.groupby("model")["success"].mean().to_dict()
print("\nBenchmark Model Capability Baseline:")
for m, acc in sorted(model_success_rates.items(), key=lambda x: x[1]):
    print(f"  {m:<40}: {acc:.1%}")

# For each prompt, determine empirical difficulty:
# 1. pass_rate: fraction of models that succeeded (0.0 = all failed, 1.0 = all passed)
# 2. min_required_capability: the capability score of the cheapest/weakest model that solved it
prompt_stats = results_long.groupby("prompt_id").agg(
    pass_rate=("success", "mean"),
    num_success=("success", "sum"),
    total_tested=("success", "count"),
)

# Compute calibrated continuous difficulty score C in [0.0, 1.0]:
# - If all models failed -> C = 1.0 (requires frontier reasoning beyond tested pool)
# - If all models passed -> C = 0.05 (simple task, easily solved by 7B models)
# - If solved only by frontier models -> C ~ 0.85
# - If solved by mid-tier models -> C ~ 0.50
def calculate_calibrated_difficulty(group):
    passed_models = group[group["success"] == 1]["model"]
    if len(passed_models) == 0:
        return 1.0  # Even GPT-4 failed
    
    # Weakest model that succeeded
    min_cap = min([model_success_rates[m] for m in passed_models])
    # Fraction of failing models weighted by model strength
    pass_fraction = len(passed_models) / len(group)
    
    # Combined difficulty index
    difficulty = 0.5 * (1.0 - pass_fraction) + 0.5 * min_cap
    return np.clip(difficulty, 0.05, 0.98)

difficulty_series = results_long.groupby("prompt_id").apply(
    calculate_calibrated_difficulty, include_groups=False
)
prompts["complexity"] = prompts["prompt_id"].map(difficulty_series)

print(f"\nComputed Prompt Complexity Target (mean={prompts['complexity'].mean():.3f}, std={prompts['complexity'].std():.3f}):")
print(prompts["complexity"].describe())

# ---------------------------------------------------------------------------
# 3. Train-Val-Test Split
# ---------------------------------------------------------------------------
X = embeddings
y = prompts["complexity"].values

train_idx, temp_idx = train_test_split(np.arange(len(X)), test_size=0.30, random_state=RANDOM_SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ---------------------------------------------------------------------------
# 4. Train Fast Ridge Regression Complexity Model
# ---------------------------------------------------------------------------
alphas = np.logspace(-3, 4, 30)
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X_train, y_train)

val_preds = np.clip(ridge_model.predict(X_val), 0.0, 1.0)
test_preds = np.clip(ridge_model.predict(X_test), 0.0, 1.0)

val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)
val_corr, _ = pearsonr(y_val, val_preds)

test_mse = mean_squared_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
test_corr, _ = pearsonr(y_test, test_preds)

print("\n" + "=" * 60)
print("COMPLEXITY REGRESSOR EVALUATION")
print("=" * 60)
print(f"Validation Set -> MSE: {val_mse:.4f}, R2: {val_r2:.4f}, Pearson r: {val_corr:.4f}")
print(f"Test Set       -> MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Pearson r: {test_corr:.4f}")
print(f"Optimal Alpha: {ridge_model.alpha_:.4f}")

# ---------------------------------------------------------------------------
# 5. Export Calibrated Artifact
# ---------------------------------------------------------------------------
artifact = {
    "coef": ridge_model.coef_,
    "intercept": ridge_model.intercept_,
    "alpha": float(ridge_model.alpha_),
    "mean_complexity": float(y_train.mean()),
    "train_size": len(X_train),
    "test_r2": float(test_r2),
    "test_corr": float(test_corr),
}

with open("gateway/complexity_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print(f"\nSaved trained complexity artifact to gateway/complexity_model.pkl")
