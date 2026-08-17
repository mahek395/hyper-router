"""
gateway/filter.py

Deterministic Constraint Pre-Filter for LLM Gateway (Stage 1).
Prunes the registered model pool in sub-millisecond time based on hard constraints
such as context window length, tools/function calling support, structured JSON mode,
modality requirements, and latency/budget SLAs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from gateway.models import ModelProfile


@dataclass
class RequestRequirements:
    """Requirements specified by the incoming API request."""
    estimated_input_tokens: int = 100
    estimated_output_tokens: int = 250
    required_features: Set[str] = field(default_factory=set)
    max_budget_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    allowed_providers: Optional[Set[str]] = None
    allowed_model_ids: Optional[Set[str]] = None
    denied_model_ids: Optional[Set[str]] = None


class ConstraintFilter:
    """
    Fast Stage 1 pre-filtering pipeline.
    Runs before any complexity or utility calculations.
    """
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token count: approx ~4 chars per token for English text/code."""
        return max(1, len(text) // 4)

    def filter(
        self,
        models: List[ModelProfile],
        requirements: RequestRequirements,
    ) -> List[ModelProfile]:
        """
        Filters candidates against hard requirements.
        Returns a non-empty subset of valid models.
        """
        valid: List[ModelProfile] = []

        for m in models:
            # 1. Deny list check
            if requirements.denied_model_ids and m.model_id in requirements.denied_model_ids:
                continue

            # 2. Allow list check
            if requirements.allowed_model_ids and m.model_id not in requirements.allowed_model_ids:
                continue

            # 3. Provider allow list
            if requirements.allowed_providers and m.provider not in requirements.allowed_providers:
                continue

            # 4. Context length check
            total_needed = requirements.estimated_input_tokens + requirements.estimated_output_tokens
            if total_needed > m.context_window:
                continue

            # 5. Required features check (tools, json_mode, vision, etc.)
            if requirements.required_features:
                if not requirements.required_features.issubset(m.features):
                    continue

            # 6. Max budget per-request check
            if requirements.max_budget_usd is not None:
                est_cost = m.estimate_cost(
                    requirements.estimated_input_tokens,
                    requirements.estimated_output_tokens
                )
                if est_cost > requirements.max_budget_usd:
                    continue

            # 7. Max latency SLA check
            if requirements.max_latency_ms is not None:
                if m.avg_latency_ms > requirements.max_latency_ms:
                    continue

            valid.append(m)

        # Fallback safeguard: if strict latency or budget filtered out all models,
        # fallback to models that satisfy functional constraints (context & features).
        if not valid and models:
            for m in models:
                total_needed = requirements.estimated_input_tokens + requirements.estimated_output_tokens
                if total_needed <= m.context_window:
                    if not requirements.required_features or requirements.required_features.issubset(m.features):
                        valid.append(m)

        return valid if valid else models
