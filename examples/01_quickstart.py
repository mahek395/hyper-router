"""
examples/01_quickstart.py

Quickstart guide for routing prompts with HyperRouter in less than 10 lines of code.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy

def main():
    # 1. Initialize router (loads default model catalog and trained complexity probe)
    router = GatewayRouter()

    # 2. Define your prompt
    prompt = "Can you help me write a Python function that uses recursion to reverse a linked list?"

    # 3. Route the prompt with a Balanced policy
    decision = router.route(
        prompt_text=prompt,
        config=RoutingConfig(policy=RoutingPolicy.BALANCED, lambda_cost=100.0)
    )

    # 4. View the routing outcome
    print("=" * 60)
    print("QUICKSTART ROUTING DECISION")
    print("=" * 60)
    print(f"Prompt            : {prompt}")
    print(f"Prompt Complexity : {decision.prompt_complexity:.3f} (0.0=easy, 1.0=expert)")
    print(f"Selected Model    : {decision.selected_model_id}")
    print(f"Estimated Cost    : ${decision.estimated_cost_usd:.6f}")
    print(f"Routing Speed     : {decision.routing_latency_ms:.3f} ms")
    print(f"Fallback Backup   : {decision.fallback_model_id}")
    print(f"Reasoning         : {decision.decision_reason}")
    print("=" * 60)


if __name__ == "__main__":
    main()
