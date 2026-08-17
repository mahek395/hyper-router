"""
gateway/server.py

FastAPI OpenAI-Compatible Gateway Server with Intelligent Dynamic Routing.
Endpoints:
  - POST /v1/route: Inspect routing decision for any prompt
  - GET  /v1/models: List registered models in registry
  - POST /v1/chat/completions: OpenAI-compatible endpoint with automatic intelligent routing
"""

import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath("."))
from gateway.models import ModelRegistry, ModelProfile
from gateway.filter import RequestRequirements
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy, RoutingDecision

app = FastAPI(
    title="HyperRouter - Intelligent LLM Gateway",
    description="Ultra-low latency, Pareto-optimal model routing engine for LLM APIs",
    version="1.0.0",
)

# Global Router Instance
router = GatewayRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class RouteRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    policy: RoutingPolicy = RoutingPolicy.BALANCED
    lambda_cost: float = 100.0
    allowed_models: Optional[List[str]] = None
    required_features: Optional[List[str]] = None
    max_budget_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto" # 'auto' triggers intelligent dynamic routing
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    routing_policy: Optional[RoutingPolicy] = RoutingPolicy.BALANCED
    lambda_cost: Optional[float] = 100.0


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "registered_models": len(router.registry.list_models())}


@app.get("/v1/models")
def list_models():
    """Lists all registered models in OpenAI-compatible format."""
    return {
        "object": "list",
        "data": [
            {
                "id": m.model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": m.provider,
                "capability_score": m.capability_score,
                "input_cost_per_m": m.input_cost_per_m,
                "output_cost_per_m": m.output_cost_per_m,
                "features": list(m.features),
            }
            for m in router.registry.list_models()
        ],
    }


@app.post("/v1/route")
def get_routing_decision(req: RouteRequest) -> Dict[str, Any]:
    """Inspects routing decision without calling the upstream LLM."""
    if req.prompt:
        text = req.prompt
    elif req.messages:
        text = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
    else:
        raise HTTPException(status_code=400, detail="Must provide 'prompt' or 'messages'")

    features = set(req.required_features) if req.required_features else set()
    requirements = RequestRequirements(
        estimated_input_tokens=len(text) // 4,
        required_features=features,
        allowed_model_ids=set(req.allowed_models) if req.allowed_models else None,
        max_budget_usd=req.max_budget_usd,
        max_latency_ms=req.max_latency_ms,
    )

    config = RoutingConfig(
        policy=req.policy,
        lambda_cost=req.lambda_cost,
    )

    decision = router.route(text, requirements=requirements, config=config)

    return {
        "selected_model": decision.selected_model_id,
        "provider": decision.selected_model_profile.provider,
        "prompt_complexity": round(decision.prompt_complexity, 4),
        "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
        "routing_latency_ms": round(decision.routing_latency_ms, 3),
        "candidates_evaluated": decision.candidates_evaluated,
        "decision_reason": decision.decision_reason,
        "fallback_model": decision.fallback_model_id,
        "all_candidate_scores": {k: round(v, 4) for k, v in decision.all_candidate_scores.items()},
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion proxy endpoint.
    If model is 'auto', dynamically routes to the Pareto-optimal LLM.
    """
    prompt_text = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
    
    required_features = set()
    if req.tools:
        required_features.add("tools")

    # If model is explicit (e.g. 'openai/gpt-4o'), use it directly
    if req.model != "auto" and req.model in [m.model_id for m in router.registry.list_models()]:
        target_model = req.model
        reason = "Direct user model specification"
        complexity = 0.5
        est_cost = 0.0
    else:
        # Intelligent Dynamic Routing
        requirements = RequestRequirements(
            estimated_input_tokens=len(prompt_text) // 4,
            estimated_output_tokens=req.max_tokens or 250,
            required_features=required_features,
        )
        config = RoutingConfig(
            policy=req.routing_policy or RoutingPolicy.BALANCED,
            lambda_cost=req.lambda_cost or 100.0,
        )
        decision = router.route(prompt_text, requirements=requirements, config=config)
        target_model = decision.selected_model_id
        reason = decision.decision_reason
        complexity = decision.prompt_complexity
        est_cost = decision.estimated_cost_usd

    # Return OpenAI-compatible response mock with routing metadata
    return {
        "id": "chatcmpl-hyperrouter-mock",
        "object": "chat.completion",
        "created": 1700000000,
        "model": target_model,
        "routing_metadata": {
            "prompt_complexity": complexity,
            "decision_reason": reason,
            "estimated_cost_usd": est_cost,
        },
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[HyperRouter Proxy]: Successfully routed prompt to '{target_model}' ({reason}).",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_text) // 4,
            "completion_tokens": 40,
            "total_tokens": (len(prompt_text) // 4) + 40,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
