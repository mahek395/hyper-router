"""
examples/03_custom_models_and_fallbacks.py

Demonstrates registering a brand-new custom self-hosted model (e.g. vLLM or Ollama)
and utilizing the automated cascade fallback recommendation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.models import ModelRegistry, ModelProfile
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy


def main():
    # 1. Create a fresh custom registry
    registry = ModelRegistry(include_defaults=False)

    # 2. Register an internal self-hosted vLLM endpoint
    registry.register(ModelProfile(
        model_id="internal-vllm/qwen-2.5-coder-32b",
        provider="vllm-internal",
        input_cost_per_m=0.10,   # Internal GPU electricity/cluster cost
        output_cost_per_m=0.15,
        capability_score=0.88,   # Strong coding benchmark rating
        context_window=64_000,
        features={"tools", "json_mode"},
        avg_latency_ms=320.0,
        description="On-premise Qwen 2.5 Coder served on 2x A100 GPUs",
    ))

    # 3. Register a cheap local fallback worker
    registry.register(ModelProfile(
        model_id="internal-vllm/llama-3.1-8b",
        provider="vllm-internal",
        input_cost_per_m=0.03,
        output_cost_per_m=0.05,
        capability_score=0.55,
        context_window=32_000,
        features={"tools", "json_mode"},
        avg_latency_ms=120.0,
        description="Local LLaMA 8B for simple queries",
    ))

    # 4. Register a cloud frontier model as the ultimate fallback
    registry.register(ModelProfile(
        model_id="anthropic/claude-3-5-sonnet",
        provider="anthropic",
        input_cost_per_m=3.00,
        output_cost_per_m=15.00,
        capability_score=0.95,
        features={"tools", "json_mode", "vision"},
        avg_latency_ms=650.0,
        description="Anthropic Claude 3.5 Sonnet for expert tasks",
    ))

    router = GatewayRouter(registry=registry)

    # Query requiring intermediate coding capability
    prompt = "Write an async Python client using httpx with exponential backoff and retry decorators."

    decision = router.route(prompt, config=RoutingConfig(policy=RoutingPolicy.BALANCED, lambda_cost=100.0))

    print("=" * 70)
    print("CUSTOM ON-PREM MODEL POOL ROUTING")
    print("=" * 70)
    print(f"Prompt         : {prompt}")
    print(f"Complexity     : {decision.prompt_complexity:.3f}")
    print(f"Primary Target : {decision.selected_model_id} (Est Cost: ${decision.estimated_cost_usd:.6f})")
    print(f"Backup Fallback: {decision.fallback_model_id}")
    print(f"Rationale      : {decision.decision_reason}")
    print("=" * 70)


if __name__ == "__main__":
    main()
