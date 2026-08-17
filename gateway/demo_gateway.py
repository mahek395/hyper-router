import os
import sys

sys.path.insert(0, os.path.abspath("."))
from gateway.models import ModelRegistry, ModelProfile, ModelFeature
from gateway.filter import RequestRequirements
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy


def run_gateway_demo():
    print("=" * 80)
    print("      PRODUCTION LLM GATEWAY ROUTING ENGINE - LIVE DEMO")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Initialize Gateway Router
    # -----------------------------------------------------------------------
    router = GatewayRouter()
    print(f"Initialized Gateway with {len(router.registry.list_models())} standard models registered.\n")

    # -----------------------------------------------------------------------
    # Scenario 1: Standard Enterprise Tenant (Balanced Policy)
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("SCENARIO 1: Enterprise Tenant (Balanced Cost & Quality Optimization)")
    print("Registered Pool: [GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash, GPT-4o-mini]")
    print("-" * 80)

    enterprise_models = {"openai/gpt-4o", "anthropic/claude-3-5-sonnet", "google/gemini-2.5-flash", "openai/gpt-4o-mini"}
    enterprise_config = RoutingConfig(policy=RoutingPolicy.BALANCED, lambda_cost=150.0)

    queries_1 = [
        "Hi, can you tell me what time zone Tokyo is in?",
        "Write a Python script using pandas and scikit-learn to train a random forest classifier and plot the ROC curve.",
        "Analyze the micro-architectural differences between Intel Lunar Lake and AMD Zen 5 with cache hierarchy diagrams.",
    ]

    for q in queries_1:
        req = RequestRequirements(
            allowed_model_ids=enterprise_models,
            estimated_input_tokens=len(q) // 4,
            estimated_output_tokens=300,
        )
        decision = router.route(q, requirements=req, config=enterprise_config)
        print(f"\nPrompt: \"{q}\"")
        print(f"  [>] Complexity Score  : {decision.prompt_complexity:.3f}")
        print(f"  [>] Routed Model      : {decision.selected_model_id}")
        print(f"  [>] Est. Cost         : ${decision.estimated_cost_usd:.6f}")
        print(f"  [>] Latency Profile   : {decision.routing_latency_ms:.2f} ms")
        print(f"  [>] Fallback Model    : {decision.fallback_model_id}")
        print(f"  [>] Decision Rationale: {decision.decision_reason}")

    # -----------------------------------------------------------------------
    # Scenario 2: Budget-Constrained Startup (Cost-Minimizing with Open Source)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 2: Startup Tenant (Aggressive Cost Minimization)")
    print("Registered Pool: [DeepSeek V3 ($0.14/M), Llama 3.3 70B ($0.35/M), Llama 3.1 8B ($0.05/M)]")
    print("-" * 80)

    startup_registry = ModelRegistry(include_defaults=False)
    startup_registry.register(ModelProfile(
        model_id="deepseek/deepseek-chat-v3",
        provider="deepseek",
        input_cost_per_m=0.14,
        output_cost_per_m=0.28,
        capability_score=0.90,
        features={"tools", "json_mode"},
        avg_latency_ms=500.0,
    ))
    startup_registry.register(ModelProfile(
        model_id="meta/llama-3.3-70b-instruct",
        provider="meta",
        input_cost_per_m=0.35,
        output_cost_per_m=0.40,
        capability_score=0.84,
        features={"tools", "json_mode"},
        avg_latency_ms=400.0,
    ))
    startup_registry.register(ModelProfile(
        model_id="meta/llama-3.1-8b-instruct",
        provider="meta",
        input_cost_per_m=0.05,
        output_cost_per_m=0.08,
        capability_score=0.55,
        features={"tools", "json_mode"},
        avg_latency_ms=180.0,
    ))

    startup_router = GatewayRouter(registry=startup_registry, complexity_scorer=router.scorer)
    startup_config = RoutingConfig(policy=RoutingPolicy.COST_MINIMIZING)

    queries_2 = [
        "Translate 'Good morning, how are you?' into Spanish, German, and Japanese.",
        "Refactor this 300-line asynchronous WebSocket handler in Rust to eliminate mutex lock contention.",
    ]

    for q in queries_2:
        decision = startup_router.route(q, config=startup_config)
        print(f"\nPrompt: \"{q}\"")
        print(f"  [>] Complexity Score  : {decision.prompt_complexity:.3f}")
        print(f"  [>] Routed Model      : {decision.selected_model_id}")
        print(f"  [>] Est. Cost         : ${decision.estimated_cost_usd:.6f}")
        print(f"  [>] Decision Rationale: {decision.decision_reason}")

    # -----------------------------------------------------------------------
    # Scenario 3: Hard Feature Constraint (Tools & JSON Schema Enforcement)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 3: Hard Feature Constraint Enforcement (Agent Tool Use)")
    print("Constraint: Request requires 'tools' & 'vision' support with max latency < 400ms")
    print("-" * 80)

    agent_req = RequestRequirements(
        estimated_input_tokens=500,
        estimated_output_tokens=200,
        required_features={"tools", "vision"},
        max_latency_ms=400.0,
    )
    agent_prompt = "Look at this image of an invoice, extract the total amount and invoke `submit_payment(amount)` tool."
    decision = router.route(agent_prompt, requirements=agent_req)
    print(f"\nPrompt: \"{agent_prompt}\"")
    print(f"  [>] Candidates Passing Stage 1 Filter: {decision.candidates_evaluated}")
    print(f"  [>] Routed Model                      : {decision.selected_model_id}")
    print(f"  [>] Est. Cost                         : ${decision.estimated_cost_usd:.6f}")
    print(f"  [>] Model Latency Profile             : {decision.selected_model_profile.avg_latency_ms:.0f} ms")
    print(f"  [>] Decision Rationale                : {decision.decision_reason}")
    print("=" * 80)


if __name__ == "__main__":
    run_gateway_demo()
