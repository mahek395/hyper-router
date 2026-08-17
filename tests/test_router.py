"""
tests/test_router.py

Unit tests for the HyperRouter LLM Gateway Routing Engine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.models import ModelRegistry, ModelProfile
from gateway.filter import ConstraintFilter, RequestRequirements
from gateway.complexity_scorer import ComplexityScorer
from gateway.router import GatewayRouter, RoutingConfig, RoutingPolicy


class TestHyperRouter(unittest.TestCase):
    def setUp(self):
        self.router = GatewayRouter()

    def test_model_registry_registration(self):
        registry = ModelRegistry(include_defaults=False)
        profile = ModelProfile(
            model_id="test/model-a",
            provider="test",
            input_cost_per_m=1.0,
            output_cost_per_m=2.0,
            capability_score=0.75,
        )
        registry.register(profile)
        self.assertEqual(len(registry.list_models()), 1)
        self.assertEqual(registry.get("test/model-a").model_id, "test/model-a")

    def test_constraint_filter_context_window(self):
        filter_engine = ConstraintFilter()
        models = [
            ModelProfile(model_id="small", provider="p", input_cost_per_m=0.1, output_cost_per_m=0.1, capability_score=0.5, context_window=4000),
            ModelProfile(model_id="large", provider="p", input_cost_per_m=1.0, output_cost_per_m=1.0, capability_score=0.9, context_window=128000),
        ]
        reqs = RequestRequirements(estimated_input_tokens=10000, estimated_output_tokens=500)
        filtered = filter_engine.filter(models, reqs)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].model_id, "large")

    def test_constraint_filter_required_features(self):
        filter_engine = ConstraintFilter()
        models = [
            ModelProfile(model_id="no-tools", provider="p", input_cost_per_m=0.1, output_cost_per_m=0.1, capability_score=0.5, features=set()),
            ModelProfile(model_id="with-tools", provider="p", input_cost_per_m=0.5, output_cost_per_m=0.5, capability_score=0.8, features={"tools", "json_mode"}),
        ]
        reqs = RequestRequirements(required_features={"tools"})
        filtered = filter_engine.filter(models, reqs)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].model_id, "with-tools")

    def test_complexity_scorer_range(self):
        scorer = ComplexityScorer()
        score = scorer.score("Hello world, what is 2 + 2?")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_routing_decision_structure(self):
        decision = self.router.route("Write a quick Python hello world script.")
        self.assertIsNotNone(decision.selected_model_id)
        self.assertGreater(decision.estimated_cost_usd, 0.0)
        self.assertGreaterEqual(decision.prompt_complexity, 0.0)
        self.assertLessEqual(decision.prompt_complexity, 1.0)
        self.assertGreater(decision.routing_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
