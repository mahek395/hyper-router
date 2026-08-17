"""
examples/02_multi_tenant_routing.py

Demonstrates how multiple tenants (customers) can each have their own
registered model pool, budget constraints, and routing policies.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.models import ModelRegistry, ModelProfile
from gateway.filter import RequestRequirements
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy


def main():
    router = GatewayRouter()

    # Define 3 different tenant configurations:
    tenants = {
        "Acme Enterprise": {
            "models": {"openai/gpt-4o", "anthropic/claude-3-5-sonnet", "google/gemini-2.5-flash"},
            "policy": RoutingPolicy.BALANCED,
            "lambda": 50.0, # Quality-biased
        },
        "Budget Startup": {
            "models": {"meta/llama-3.1-8b-instruct", "deepseek/deepseek-chat-v3", "openai/gpt-4o-mini"},
            "policy": RoutingPolicy.COST_MINIMIZING,
            "lambda": 500.0, # Cost-biased
        },
        "Fast SLA Bot": {
            "models": {"mistralai/mistral-7b-instruct", "openai/gpt-4o-mini", "google/gemini-2.5-flash"},
            "policy": RoutingPolicy.LATENCY_MINIMIZING,
            "lambda": 100.0,
        },
    }

    test_queries = [
        "What is the capital of Canada?",
        "Write a complete Dockerfile and Kubernetes deployment manifest for a FastAPI service with Redis.",
    ]

    for tenant_name, cfg in tenants.items():
        print("\n" + "=" * 70)
        print(f"TENANT: {tenant_name} (Policy: {cfg['policy'].value}, Models: {len(cfg['models'])})")
        print("=" * 70)

        reqs = RequestRequirements(allowed_model_ids=cfg["models"])
        config = RoutingConfig(policy=cfg["policy"], lambda_cost=cfg["lambda"])

        for q in test_queries:
            decision = router.route(q, requirements=reqs, config=config)
            print(f"\nQuery: \"{q}\"")
            print(f"  -> Complexity Score: {decision.prompt_complexity:.3f}")
            print(f"  -> Chosen Model    : {decision.selected_model_id}")
            print(f"  -> Est. Cost       : ${decision.estimated_cost_usd:.6f}")
            print(f"  -> Why Chosen      : {decision.decision_reason}")


if __name__ == "__main__":
    main()
