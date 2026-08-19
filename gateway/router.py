"""
gateway/router.py

Core LLM Gateway Routing Engine.

Dynamic routing across whatever models are registered in PostgreSQL.
Optimizes capability, cost, and latency according to the selected policy.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from gateway.models import ModelProfile, ModelRegistry
from gateway.filter import ConstraintFilter, RequestRequirements
from gateway.complexity_scorer import ComplexityScorer


class RoutingPolicy(str, Enum):
    BALANCED = "balanced"
    COST_MINIMIZING = "cost_minimizing"
    QUALITY_MAXIMIZING = "quality_maximizing"
    THRESHOLD_CASCADE = "threshold_cascade"
    LATENCY_MINIMIZING = "latency_minimizing"


@dataclass
class RoutingConfig:
    """Configuration for a routing decision."""

    policy: RoutingPolicy = RoutingPolicy.BALANCED

    # Cost penalty used by BALANCED mode.
    lambda_cost: float = 100.0

    # Complexity threshold for THRESHOLD_CASCADE.
    cascade_threshold: float = 0.60

    # How far below prompt complexity a model may be and still qualify.
    capability_margin: float = 0.08

    # Absolute minimum capability allowed.
    min_capability_floor: float = 0.0

    # Whether a fallback/escalation model should be recommended.
    enable_cascade_fallback: bool = True


@dataclass
class RoutingDecision:
    """Complete routing decision."""

    selected_model_id: str
    selected_model_profile: ModelProfile

    estimated_cost_usd: float
    prompt_complexity: float
    routing_latency_ms: float

    candidates_evaluated: int
    decision_reason: str

    fallback_model_id: Optional[str] = None

    all_candidate_scores: Dict[str, float] = field(default_factory=dict)


class GatewayRouter:
    """
    Production-grade dynamic LLM router.

    Model discovery comes entirely from ModelRegistry.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        complexity_scorer: Optional[ComplexityScorer] = None,
        constraint_filter: Optional[ConstraintFilter] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.scorer = complexity_scorer or ComplexityScorer()
        self.filter = constraint_filter or ConstraintFilter()

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    def route(
        self,
        prompt_text: str,
        requirements: Optional[RequestRequirements] = None,
        config: Optional[RoutingConfig] = None,
        embedding: Optional[Any] = None,
    ) -> RoutingDecision:

        start_time = time.perf_counter()

        if requirements is None:
            est_tokens = self.filter.estimate_tokens(prompt_text)

            requirements = RequestRequirements(
                estimated_input_tokens=est_tokens
            )

        if config is None:
            config = RoutingConfig()

        # --------------------------------------------------------------
        # Stage 1: deterministic filtering
        # --------------------------------------------------------------

        available_models = self.registry.list_models()

        if requirements.allowed_model_ids:
            available_models = [
                m
                for m in available_models
                if m.model_id in requirements.allowed_model_ids
            ]

        candidates = self.filter.filter(
            available_models,
            requirements,
        )

        if not candidates:
            raise ValueError(
                "No models in registry meet the request requirements."
            )

        # --------------------------------------------------------------
        # Stage 2: complexity estimation
        # --------------------------------------------------------------

        complexity = self.scorer.score(
            prompt_text,
            embedding=embedding,
        )

        # --------------------------------------------------------------
        # Stage 3: model selection
        # --------------------------------------------------------------

        (
            chosen_profile,
            fallback_profile,
            scores,
            reason,
        ) = self._select_model(
            candidates=candidates,
            complexity=complexity,
            requirements=requirements,
            config=config,
        )

        # --------------------------------------------------------------
        # Stage 4: cost estimation
        # --------------------------------------------------------------

        cost_est = chosen_profile.estimate_cost(
            requirements.estimated_input_tokens,
            requirements.estimated_output_tokens,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return RoutingDecision(
            selected_model_id=chosen_profile.model_id,
            selected_model_profile=chosen_profile,

            estimated_cost_usd=cost_est,
            prompt_complexity=complexity,
            routing_latency_ms=elapsed_ms,

            candidates_evaluated=len(candidates),
            decision_reason=reason,

            fallback_model_id=(
                fallback_profile.model_id
                if fallback_profile
                else None
            ),

            all_candidate_scores=scores,
        )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _select_model(
        self,
        candidates: List[ModelProfile],
        complexity: float,
        requirements: RequestRequirements,
        config: RoutingConfig,
    ) -> Tuple[
        ModelProfile,
        Optional[ModelProfile],
        Dict[str, float],
        str,
    ]:

        scores: Dict[str, float] = {}

        in_tok = requirements.estimated_input_tokens
        out_tok = requirements.estimated_output_tokens

        # Highest-capability model among available candidates.
        sorted_by_capability = sorted(
            candidates,
            key=lambda m: m.capability_score,
            reverse=True,
        )

        frontier_model = sorted_by_capability[0]

        # Capability required to satisfy the request.
        required_capability = max(
            config.min_capability_floor,
            complexity - config.capability_margin,
        )

        # Models that are actually capable enough for the request.
        qualified = [
            m
            for m in candidates
            if m.capability_score >= required_capability
        ]

        # --------------------------------------------------------------
        # THRESHOLD CASCADE
        # --------------------------------------------------------------

        if config.policy == RoutingPolicy.THRESHOLD_CASCADE:

            if complexity < config.cascade_threshold:

                # For easy prompts, choose the cheapest model
                # that can still satisfy the complexity requirement.
                if qualified:
                    chosen = min(
                        qualified,
                        key=lambda m: m.estimate_cost(
                            in_tok,
                            out_tok,
                        ),
                    )

                    reason = (
                        "Threshold Cascade: "
                        f"Low complexity ({complexity:.2f} < "
                        f"{config.cascade_threshold:.2f}), "
                        f"selected cheapest qualified model "
                        f"({chosen.model_id})"
                    )

                else:
                    # No model satisfies capability requirement.
                    # Do NOT choose the cheapest model.
                    chosen = frontier_model

                    reason = (
                        "Threshold Cascade: "
                        f"Low complexity ({complexity:.2f} < "
                        f"{config.cascade_threshold:.2f}), "
                        "but no model met the capability requirement; "
                        f"selected strongest available model "
                        f"({chosen.model_id})"
                    )

            else:

                # High complexity → strongest model.
                chosen = frontier_model

                reason = (
                    "Threshold Cascade: "
                    f"High complexity ({complexity:.2f} >= "
                    f"{config.cascade_threshold:.2f}), "
                    f"selected frontier model "
                    f"({chosen.model_id})"
                )

            fallback = (
                frontier_model
                if (
                    config.enable_cascade_fallback
                    and chosen.model_id != frontier_model.model_id
                )
                else None
            )

            return (
                chosen,
                fallback,
                {
                    m.model_id: m.capability_score
                    for m in candidates
                },
                reason,
            )

        # --------------------------------------------------------------
        # COST MINIMIZING
        # --------------------------------------------------------------

        if config.policy == RoutingPolicy.COST_MINIMIZING:

            if qualified:

                # Correct cost minimization:
                # cheapest model among models that are capable enough.
                chosen = min(
                    qualified,
                    key=lambda m: m.estimate_cost(
                        in_tok,
                        out_tok,
                    ),
                )

                reason = (
                    "Cost Minimizing: "
                    f"Selected cheapest qualified model "
                    f"({chosen.model_id}) with capability "
                    f"{chosen.capability_score:.2f} "
                    f"for complexity {complexity:.2f}"
                )

            else:

                # IMPORTANT:
                # No model is qualified.
                #
                # Previous implementation fell back to ALL models
                # and selected the cheapest. That could send a highly
                # complex request to a clearly under-capable model.
                #
                # We now select the strongest available model.
                chosen = frontier_model

                reason = (
                    "Cost Minimizing: "
                    f"No model met required capability "
                    f"{required_capability:.2f} for complexity "
                    f"{complexity:.2f}; "
                    f"selected strongest available model "
                    f"({chosen.model_id}, capability="
                    f"{chosen.capability_score:.2f})"
                )

            fallback = (
                frontier_model
                if (
                    config.enable_cascade_fallback
                    and chosen.model_id != frontier_model.model_id
                )
                else None
            )

            return (
                chosen,
                fallback,
                {
                    m.model_id: -m.estimate_cost(
                        in_tok,
                        out_tok,
                    )
                    for m in candidates
                },
                reason,
            )

        # --------------------------------------------------------------
        # QUALITY MAXIMIZING
        # --------------------------------------------------------------

        if config.policy == RoutingPolicy.QUALITY_MAXIMIZING:

            chosen = frontier_model

            reason = (
                "Quality Maximizing: "
                f"Selected highest-capability model "
                f"({chosen.model_id}, "
                f"capability={chosen.capability_score:.2f})"
            )

            return (
                chosen,
                None,
                {
                    m.model_id: m.capability_score
                    for m in candidates
                },
                reason,
            )

        # --------------------------------------------------------------
        # LATENCY MINIMIZING
        # --------------------------------------------------------------

        if config.policy == RoutingPolicy.LATENCY_MINIMIZING:

            if qualified:

                chosen = min(
                    qualified,
                    key=lambda m: m.avg_latency_ms,
                )

                reason = (
                    "Latency Minimizing: "
                    f"Selected fastest qualified model "
                    f"({chosen.model_id}, "
                    f"{chosen.avg_latency_ms:.0f}ms)"
                )

            else:

                # If no model is capable enough, don't choose a
                # potentially inadequate fast model.
                chosen = frontier_model

                reason = (
                    "Latency Minimizing: "
                    f"No model met required capability "
                    f"{required_capability:.2f}; "
                    f"selected strongest available model "
                    f"({chosen.model_id})"
                )

            fallback = (
                frontier_model
                if (
                    config.enable_cascade_fallback
                    and chosen.model_id != frontier_model.model_id
                )
                else None
            )

            return (
                chosen,
                fallback,
                {
                    m.model_id: -m.avg_latency_ms
                    for m in candidates
                },
                reason,
            )

        # --------------------------------------------------------------
        # BALANCED
        # --------------------------------------------------------------

        for model in candidates:

            diff = (
                model.capability_score
                - complexity
            )

            if diff >= 0:

                # Model is capable enough.
                capability_fit = (
                    1.0
                    + 0.15 * diff
                )

            else:

                # Model is under-qualified.
                # Penalty grows quadratically.
                capability_fit = max(
                    0.0,
                    1.0 + 2.5 * diff,
                )

            cost = model.estimate_cost(
                in_tok,
                out_tok,
            )

            utility = (
                capability_fit
                - config.lambda_cost * cost
            )

            scores[model.model_id] = float(
                utility
            )

        best_model_id = max(
            scores,
            key=scores.get,
        )

        chosen = (
            self.registry.get(best_model_id)
            or candidates[0]
        )

        fallback = (
            frontier_model
            if (
                config.enable_cascade_fallback
                and chosen.model_id
                != frontier_model.model_id
            )
            else None
        )

        reason = (
            "Balanced Utility "
            f"(lambda={config.lambda_cost:.1f}): "
            f"Complexity {complexity:.2f} -> "
            f"Selected {chosen.model_id} "
            f"(Capability={chosen.capability_score:.2f}, "
            f"Est Cost="
            f"${chosen.estimate_cost(in_tok, out_tok):.6f})"
        )

        return (
            chosen,
            fallback,
            scores,
            reason,
        )