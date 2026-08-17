"""
gateway/router.py

Core LLM Gateway Routing Engine (Stage 3 & 4).
Decoupled, dynamic, and multi-tenant routing engine that optimizes cost vs quality
across whatever models the user has registered on the gateway.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

from gateway.models import ModelProfile, ModelRegistry
from gateway.filter import ConstraintFilter, RequestRequirements
from gateway.complexity_scorer import ComplexityScorer


class RoutingPolicy(str, Enum):
    BALANCED = "balanced"                    # Fit(Rm, C) - lambda * Cost
    COST_MINIMIZING = "cost_minimizing"      # Cheapest model with Rm >= C - margin
    QUALITY_MAXIMIZING = "quality_maximizing"# Highest capability within budget
    THRESHOLD_CASCADE = "threshold_cascade"  # Cheap model if C < theta else Frontier
    LATENCY_MINIMIZING = "latency_minimizing"# Fastest model with Rm >= C


@dataclass
class RoutingConfig:
    """Configures the routing policy and parameters for a request or tenant."""
    policy: RoutingPolicy = RoutingPolicy.BALANCED
    lambda_cost: float = 100.0               # Cost trade-off coefficient
    cascade_threshold: float = 0.60          # Complexity cutoff for cheap vs frontier
    capability_margin: float = 0.08          # Allowed margin below complexity
    min_capability_floor: float = 0.0        # Hard lower bound on capability
    enable_cascade_fallback: bool = True     # Include recommended escalation model


@dataclass
class RoutingDecision:
    """Complete structured output from the routing engine."""
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
    Production-grade LLM Gateway Router.
    """
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        complexity_scorer: Optional[ComplexityScorer] = None,
        constraint_filter: Optional[ConstraintFilter] = None,
    ):
        self.registry = registry or ModelRegistry(include_defaults=True)
        self.scorer = complexity_scorer or ComplexityScorer()
        self.filter = constraint_filter or ConstraintFilter()

    def route(
        self,
        prompt_text: str,
        requirements: Optional[RequestRequirements] = None,
        config: Optional[RoutingConfig] = None,
        embedding: Optional[Any] = None,
    ) -> RoutingDecision:
        """
        Main entrypoint: Routes an incoming prompt to the best available LLM.
        
        Args:
            prompt_text: Raw incoming user prompt / messages
            requirements: Constraint requirements (context length, tools, JSON mode, budget)
            config: Routing policy parameters (Balanced, Cost, Quality, Cascade)
            embedding: Optional precomputed vector for zero-embedder latency
            
        Returns:
            RoutingDecision object with selected model and execution metadata.
        """
        start_time = time.perf_counter()

        if requirements is None:
            est_tokens = self.filter.estimate_tokens(prompt_text)
            requirements = RequestRequirements(estimated_input_tokens=est_tokens)

        if config is None:
            config = RoutingConfig()

        # -------------------------------------------------------------------
        # Stage 1: Deterministic Constraint Filtering
        # -------------------------------------------------------------------
        available_models = self.registry.list_models()
        if requirements.allowed_model_ids:
            available_models = [m for m in available_models if m.model_id in requirements.allowed_model_ids]

        candidates = self.filter.filter(available_models, requirements)
        if not candidates:
            raise ValueError("No models in registry meet the request requirements.")

        # -------------------------------------------------------------------
        # Stage 2: Complexity Estimation
        # -------------------------------------------------------------------
        complexity = self.scorer.score(prompt_text, embedding=embedding)

        # -------------------------------------------------------------------
        # Stage 3: Dynamic Utility Selection
        # -------------------------------------------------------------------
        chosen_profile, fallback_profile, scores, reason = self._select_model(
            candidates=candidates,
            complexity=complexity,
            requirements=requirements,
            config=config,
        )

        cost_est = chosen_profile.estimate_cost(
            requirements.estimated_input_tokens,
            requirements.estimated_output_tokens,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RoutingDecision(
            selected_model_id=chosen_profile.model_id,
            selected_model_profile=chosen_profile,
            estimated_cost_usd=cost_est,
            prompt_complexity=complexity,
            routing_latency_ms=elapsed_ms,
            candidates_evaluated=len(candidates),
            decision_reason=reason,
            fallback_model_id=fallback_profile.model_id if fallback_profile else None,
            all_candidate_scores=scores,
        )

    def _select_model(
        self,
        candidates: List[ModelProfile],
        complexity: float,
        requirements: RequestRequirements,
        config: RoutingConfig,
    ) -> Tuple[ModelProfile, Optional[ModelProfile], Dict[str, float], str]:
        """Calculates policy scores across all candidates and picks optimal model."""
        scores: Dict[str, float] = {}
        in_tok = requirements.estimated_input_tokens
        out_tok = requirements.estimated_output_tokens

        # Identify candidate models sorted by capability
        sorted_by_cap = sorted(candidates, key=lambda m: m.capability_score, reverse=True)
        frontier_model = sorted_by_cap[0]

        if config.policy == RoutingPolicy.THRESHOLD_CASCADE:
            # Simple & robust cascade: if complexity < threshold, pick cheapest qualified model
            if complexity < config.cascade_threshold:
                # Filter models capable enough for this lower complexity
                qualified = [m for m in candidates if m.capability_score >= (complexity - config.capability_margin)]
                if not qualified:
                    qualified = candidates
                chosen = min(qualified, key=lambda m: m.estimate_cost(in_tok, out_tok))
                reason = f"Threshold Cascade: Low complexity ({complexity:.2f} < {config.cascade_threshold:.2f}), routed to cost-efficient {chosen.model_id}"
            else:
                chosen = frontier_model
                reason = f"Threshold Cascade: High complexity ({complexity:.2f} >= {config.cascade_threshold:.2f}), routed to frontier {chosen.model_id}"

            fallback = frontier_model if chosen.model_id != frontier_model.model_id else None
            return chosen, fallback, {m.model_id: m.capability_score for m in candidates}, reason

        elif config.policy == RoutingPolicy.COST_MINIMIZING:
            qualified = [m for m in candidates if m.capability_score >= (complexity - config.capability_margin)]
            if not qualified:
                qualified = candidates
            chosen = min(qualified, key=lambda m: m.estimate_cost(in_tok, out_tok))
            fallback = frontier_model if chosen.model_id != frontier_model.model_id else None
            reason = f"Cost Minimizing: Selected cheapest qualified model ({chosen.model_id}) with capability {chosen.capability_score:.2f} for complexity {complexity:.2f}"
            return chosen, fallback, {m.model_id: -m.estimate_cost(in_tok, out_tok) for m in candidates}, reason

        elif config.policy == RoutingPolicy.QUALITY_MAXIMIZING:
            chosen = frontier_model
            reason = f"Quality Maximizing: Routed to highest capability model ({chosen.model_id}, rating={chosen.capability_score:.2f})"
            return chosen, None, {m.model_id: m.capability_score for m in candidates}, reason

        elif config.policy == RoutingPolicy.LATENCY_MINIMIZING:
            qualified = [m for m in candidates if m.capability_score >= (complexity - config.capability_margin)]
            if not qualified:
                qualified = candidates
            chosen = min(qualified, key=lambda m: m.avg_latency_ms)
            fallback = frontier_model if chosen.model_id != frontier_model.model_id else None
            reason = f"Latency Minimizing: Selected fastest qualified model ({chosen.model_id}, {chosen.avg_latency_ms:.0f}ms)"
            return chosen, fallback, {m.model_id: -m.avg_latency_ms for m in candidates}, reason

        else: # RoutingPolicy.BALANCED
            # Continuous Pareto Utility:
            # Expected Success / Capability Fit P(m) is modeled via logistic/smooth saturation:
            # If Rm >= C: high success probability (near 1.0)
            # If Rm < C: steep capability deficit penalty
            for m in candidates:
                diff = m.capability_score - complexity
                if diff >= 0:
                    # Model exceeds requirement: diminishing returns on extra capability
                    capability_fit = 1.0 + 0.15 * diff
                else:
                    # Model under-qualified: penalty quadratic with deficit
                    capability_fit = max(0.0, 1.0 + 2.5 * diff)

                cost = m.estimate_cost(in_tok, out_tok)
                utility = capability_fit - (config.lambda_cost * cost)
                scores[m.model_id] = float(utility)

            best_model_id = max(scores, key=scores.get)
            chosen = self.registry.get(best_model_id) or candidates[0]
            fallback = frontier_model if chosen.model_id != frontier_model.model_id else None
            reason = f"Balanced Utility (lambda={config.lambda_cost:.1f}): Complexity {complexity:.2f} -> Selected {chosen.model_id} (Capability={chosen.capability_score:.2f}, Est Cost=${chosen.estimate_cost(in_tok, out_tok):.6f})"
            return chosen, fallback, scores, reason
