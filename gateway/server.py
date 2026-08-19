"""
gateway/server.py

FastAPI OpenAI-Compatible Gateway Server with Intelligent Dynamic Routing.

Endpoints:
  - GET  /health
  - GET  /v1/models
  - POST /v1/route
  - POST /internal/models/reload
  - POST /v1/chat/completions

Model registration is owned by the Node admin gateway.
Python reads registered_models from PostgreSQL through ModelRegistry.
"""

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath("."))

from gateway.models import ModelRegistry
from gateway.filter import RequestRequirements
from gateway.router import (
    GatewayRouter,
    RoutingConfig,
    RoutingPolicy,
)

app = FastAPI(
    title="HyperRouter - Intelligent LLM Gateway",
    description=(
        "Ultra-low latency, dynamic LLM routing engine "
        "for registered models"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Global Router Instance
# ---------------------------------------------------------------------------

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

    # Main routing policy
    policy: RoutingPolicy = RoutingPolicy.BALANCED

    # Cost sensitivity
    lambda_cost: float = 100.0

    # Optional request constraints
    allowed_models: Optional[List[str]] = None
    required_features: Optional[List[str]] = None
    max_budget_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Dynamic routing configuration
    cascade_threshold: float = 0.60
    capability_margin: float = 0.08
    min_capability_floor: float = 0.0
    enable_cascade_fallback: bool = True


class ChatCompletionRequest(BaseModel):
    """
    Python-side OpenAI-compatible request.

    The production Node gateway normally performs the actual
    provider call. This endpoint exists for direct sidecar usage
    and local testing.
    """

    model: str = "auto"
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

    tools: Optional[List[Dict[str, Any]]] = None

    routing_policy: Optional[RoutingPolicy] = (
        RoutingPolicy.BALANCED
    )

    lambda_cost: Optional[float] = 100.0

    cascade_threshold: Optional[float] = 0.60
    capability_margin: Optional[float] = 0.08
    min_capability_floor: Optional[float] = 0.0
    enable_cascade_fallback: Optional[bool] = True


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "registered_models": len(
            router.registry.list_models()
        ),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
def list_models():
    """
    Lists models currently loaded by Python ModelRegistry.

    The source of truth is PostgreSQL registered_models.
    Python does NOT manage provider API keys.
    """

    return {
        "object": "list",
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": model.provider,
                "capability_score": model.capability_score,
                "input_cost_per_m": model.input_cost_per_m,
                "output_cost_per_m": model.output_cost_per_m,
                "context_window": model.context_window,
                "max_output_tokens": model.max_output_tokens,
                "features": list(model.features),
                "avg_latency_ms": model.avg_latency_ms,
                "description": model.description,
            }
            for model in router.registry.list_models()
        ],
    }


# ---------------------------------------------------------------------------
# Runtime model reload
# ---------------------------------------------------------------------------

@app.post("/internal/models/reload")
def reload_models():
    """
    Called by Node after a registered model is created,
    updated, or deleted.

    Keep this endpoint private to the internal network.
    """

    router.registry.refresh()

    return {
        "reloaded": True,
        "total_models": len(
            router.registry.list_models()
        ),
    }


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

@app.post("/v1/route")
def get_routing_decision(
    req: RouteRequest,
) -> Dict[str, Any]:
    """
    Inspects a routing decision without calling an
    upstream LLM provider.
    """

    # --------------------------------------------------------------
    # Build routing text
    # --------------------------------------------------------------

    if req.prompt:
        text = req.prompt

    elif req.messages:
        text = "\n".join(
            f"{message.role}: {message.content}"
            for message in req.messages
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Must provide 'prompt' "
                "or 'messages'"
            ),
        )

    # --------------------------------------------------------------
    # Required features
    # --------------------------------------------------------------

    features = (
        set(req.required_features)
        if req.required_features
        else set()
    )

    # --------------------------------------------------------------
    # Request constraints
    # --------------------------------------------------------------

    requirements = RequestRequirements(
        estimated_input_tokens=max(
            1,
            len(text) // 4,
        ),
        required_features=features,
        allowed_model_ids=(
            set(req.allowed_models)
            if req.allowed_models
            else None
        ),
        max_budget_usd=req.max_budget_usd,
        max_latency_ms=req.max_latency_ms,
    )

    # --------------------------------------------------------------
    # Dynamic routing configuration
    # --------------------------------------------------------------

    config = RoutingConfig(
        policy=req.policy,
        lambda_cost=req.lambda_cost,
        cascade_threshold=req.cascade_threshold,
        capability_margin=req.capability_margin,
        min_capability_floor=req.min_capability_floor,
        enable_cascade_fallback=(
            req.enable_cascade_fallback
        ),
    )

    # --------------------------------------------------------------
    # Route
    # --------------------------------------------------------------

    try:
        decision = router.route(
            text,
            requirements=requirements,
            config=config,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    # --------------------------------------------------------------
    # Return routing metadata
    # --------------------------------------------------------------

    return {
        "selected_model": decision.selected_model_id,
        "provider": (
            decision.selected_model_profile.provider
        ),

        "prompt_complexity": round(
            decision.prompt_complexity,
            4,
        ),

        "estimated_cost_usd": round(
            decision.estimated_cost_usd,
            6,
        ),

        "routing_latency_ms": round(
            decision.routing_latency_ms,
            3,
        ),

        "candidates_evaluated":
            decision.candidates_evaluated,

        "decision_reason":
            decision.decision_reason,

        "fallback_model":
            decision.fallback_model_id,

        "all_candidate_scores": {
            model_id: round(score, 4)
            for model_id, score
            in decision.all_candidate_scores.items()
        },
    }


# ---------------------------------------------------------------------------
# Direct Python chat-completion endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
):
    """
    Python-side OpenAI-compatible routing endpoint.

    In the main architecture:
        Client
          ↓
        Node gateway
          ↓
        Python /v1/route
          ↓
        Node generic provider caller

    Therefore this endpoint currently returns a routing/mock
    response rather than directly calling an upstream provider.
    """

    # --------------------------------------------------------------
    # Build prompt text
    # --------------------------------------------------------------

    prompt_text = "\n".join(
        f"{message.role}: {message.content}"
        for message in req.messages
    )

    # --------------------------------------------------------------
    # Required features
    # --------------------------------------------------------------

    required_features = set()

    if req.tools:
        required_features.add("tools")

    # --------------------------------------------------------------
    # Explicit model
    # --------------------------------------------------------------

    registered_model_ids = {
        model.model_id
        for model in router.registry.list_models()
    }

    if (
        req.model != "auto"
        and req.model in registered_model_ids
    ):
        target_model = req.model
        reason = (
            "Direct user model specification"
        )
        complexity = 0.5
        estimated_cost = 0.0

    else:
        # ----------------------------------------------------------
        # Automatic routing
        # ----------------------------------------------------------

        requirements = RequestRequirements(
            estimated_input_tokens=max(
                1,
                len(prompt_text) // 4,
            ),
            estimated_output_tokens=(
                req.max_tokens or 250
            ),
            required_features=required_features,
        )

        config = RoutingConfig(
            policy=(
                req.routing_policy
                or RoutingPolicy.BALANCED
            ),
            lambda_cost=(
                req.lambda_cost
                if req.lambda_cost is not None
                else 100.0
            ),
            cascade_threshold=(
                req.cascade_threshold
                if req.cascade_threshold
                is not None
                else 0.60
            ),
            capability_margin=(
                req.capability_margin
                if req.capability_margin
                is not None
                else 0.08
            ),
            min_capability_floor=(
                req.min_capability_floor
                if req.min_capability_floor
                is not None
                else 0.0
            ),
            enable_cascade_fallback=(
                req.enable_cascade_fallback
                if req.enable_cascade_fallback
                is not None
                else True
            ),
        )

        try:
            decision = router.route(
                prompt_text,
                requirements=requirements,
                config=config,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            )

        target_model = decision.selected_model_id
        reason = decision.decision_reason
        complexity = decision.prompt_complexity
        estimated_cost = decision.estimated_cost_usd

    # --------------------------------------------------------------
    # Return routing/mock response
    # --------------------------------------------------------------

    prompt_tokens = max(
        1,
        len(prompt_text) // 4,
    )

    completion_tokens = 40

    return {
        "id": "chatcmpl-hyperrouter-mock",
        "object": "chat.completion",
        "created": 1700000000,

        "model": target_model,

        "routing_metadata": {
            "prompt_complexity": complexity,
            "decision_reason": reason,
            "estimated_cost_usd": estimated_cost,
        },

        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "[HyperRouter Proxy]: "
                        f"Successfully routed prompt "
                        f"to '{target_model}' "
                        f"({reason})."
                    ),
                },
                "finish_reason": "stop",
            }
        ],

        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens
                + completion_tokens
            ),
        },
    }


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )