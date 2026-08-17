"""
gateway/models.py

Dynamic Model Registry and Capability Profile for LLM Gateway.
Decouples prompt routing from hardcoded model names, allowing users and
tenants to register any arbitrary set of models, pricing tiers, and capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class ModelFeature(str, Enum):
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"
    REASONING = "reasoning"


@dataclass
class ModelProfile:
    """
    Profile representing an LLM available in the gateway.
    
    Attributes:
        model_id: Unique identifier / API route name (e.g. 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')
        provider: Provider name ('openai', 'anthropic', 'google', 'deepseek', 'meta', 'vllm', 'custom')
        input_cost_per_m: Price in USD per 1M input tokens
        output_cost_per_m: Price in USD per 1M output tokens
        capability_score: Normalized benchmark / Arena score from 0.0 (weakest) to 1.0 (frontier)
        context_window: Maximum context length in tokens
        max_output_tokens: Maximum tokens in response
        features: Set of supported capabilities (tools, json_mode, vision, etc.)
        avg_latency_ms: Typical time-to-first-token + processing latency in ms
        description: Human-readable description
    """
    model_id: str
    provider: str
    input_cost_per_m: float
    output_cost_per_m: float
    capability_score: float
    context_window: int = 128_000
    max_output_tokens: int = 4096
    features: Set[str] = field(default_factory=lambda: {ModelFeature.TOOLS.value, ModelFeature.JSON_MODE.value})
    avg_latency_ms: float = 600.0
    description: str = ""

    def estimate_cost(self, input_tokens: int, estimated_output_tokens: int = 250) -> float:
        """Calculates expected request cost in USD."""
        cost_in = (input_tokens / 1_000_000.0) * self.input_cost_per_m
        cost_out = (estimated_output_tokens / 1_000_000.0) * self.output_cost_per_m
        return cost_in + cost_out


class ModelRegistry:
    """
    Manages registered models for a gateway instance or tenant.
    Allows dynamic registration, unregistration, and lookup.
    """
    def __init__(self, include_defaults: bool = True):
        self._models: Dict[str, ModelProfile] = {}
        if include_defaults:
            self._load_default_catalog()

    def register(self, profile: ModelProfile) -> None:
        """Register or update a model profile."""
        self._models[profile.model_id] = profile

    def unregister(self, model_id: str) -> Optional[ModelProfile]:
        """Remove a model from the registry."""
        return self._models.pop(model_id, None)

    def get(self, model_id: str) -> Optional[ModelProfile]:
        """Retrieve a model profile by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[ModelProfile]:
        """Return all currently registered models."""
        return list(self._models.values())

    def filter_by_ids(self, model_ids: List[str]) -> "ModelRegistry":
        """Create a sub-registry with a specific subset of models."""
        sub = ModelRegistry(include_defaults=False)
        for mid in model_ids:
            if mid in self._models:
                sub.register(self._models[mid])
        return sub

    def _load_default_catalog(self) -> None:
        """Pre-populate with standard frontier, mid-tier, and open-source models."""
        catalog = [
            # --- Frontier Tier (0.85 - 1.00) ---
            ModelProfile(
                model_id="openai/gpt-4o",
                provider="openai",
                input_cost_per_m=2.50,
                output_cost_per_m=10.00,
                capability_score=0.92,
                context_window=128_000,
                features={"tools", "json_mode", "vision", "streaming"},
                avg_latency_ms=550.0,
                description="OpenAI GPT-4o multimodal flagship",
            ),
            ModelProfile(
                model_id="anthropic/claude-3-5-sonnet",
                provider="anthropic",
                input_cost_per_m=3.00,
                output_cost_per_m=15.00,
                capability_score=0.95,
                context_window=200_000,
                features={"tools", "json_mode", "vision", "streaming"},
                avg_latency_ms=650.0,
                description="Anthropic Claude 3.5 Sonnet frontier reasoning and coding",
            ),
            ModelProfile(
                model_id="google/gemini-2.5-pro",
                provider="google",
                input_cost_per_m=1.25,
                output_cost_per_m=10.00,
                capability_score=0.93,
                context_window=1_000_000,
                features={"tools", "json_mode", "vision", "audio", "streaming"},
                avg_latency_ms=700.0,
                description="Google Gemini 2.5 Pro 1M context",
            ),
            ModelProfile(
                model_id="deepseek/deepseek-reasoner-r1",
                provider="deepseek",
                input_cost_per_m=0.55,
                output_cost_per_m=2.19,
                capability_score=0.96,
                context_window=64_000,
                features={"reasoning", "streaming"},
                avg_latency_ms=1800.0,
                description="DeepSeek R1 reasoning model",
            ),
            ModelProfile(
                model_id="deepseek/deepseek-chat-v3",
                provider="deepseek",
                input_cost_per_m=0.14,
                output_cost_per_m=0.28,
                capability_score=0.90,
                context_window=64_000,
                features={"tools", "json_mode", "streaming"},
                avg_latency_ms=500.0,
                description="DeepSeek V3 671B MoE",
            ),

            # --- High Performance Mid-Tier (0.70 - 0.85) ---
            ModelProfile(
                model_id="openai/gpt-4o-mini",
                provider="openai",
                input_cost_per_m=0.15,
                output_cost_per_m=0.60,
                capability_score=0.78,
                context_window=128_000,
                features={"tools", "json_mode", "vision", "streaming"},
                avg_latency_ms=280.0,
                description="OpenAI fast efficient model",
            ),
            ModelProfile(
                model_id="anthropic/claude-3-5-haiku",
                provider="anthropic",
                input_cost_per_m=0.80,
                output_cost_per_m=4.00,
                capability_score=0.82,
                context_window=200_000,
                features={"tools", "json_mode", "vision", "streaming"},
                avg_latency_ms=250.0,
                description="Anthropic high speed intelligent model",
            ),
            ModelProfile(
                model_id="google/gemini-2.5-flash",
                provider="google",
                input_cost_per_m=0.15,
                output_cost_per_m=0.60,
                capability_score=0.80,
                context_window=1_000_000,
                features={"tools", "json_mode", "vision", "audio", "streaming"},
                avg_latency_ms=260.0,
                description="Google ultra-fast multimodal flash",
            ),
            ModelProfile(
                model_id="meta/llama-3.3-70b-instruct",
                provider="meta",
                input_cost_per_m=0.35,
                output_cost_per_m=0.40,
                capability_score=0.84,
                context_window=128_000,
                features={"tools", "json_mode", "streaming"},
                avg_latency_ms=400.0,
                description="Meta Llama 3.3 70B flagship open weights",
            ),
            ModelProfile(
                model_id="qwen/qwen-2.5-72b-instruct",
                provider="qwen",
                input_cost_per_m=0.35,
                output_cost_per_m=0.40,
                capability_score=0.85,
                context_window=128_000,
                features={"tools", "json_mode", "streaming"},
                avg_latency_ms=420.0,
                description="Alibaba Qwen 2.5 72B Instruct",
            ),

            # --- Budget & Fast Worker Tier (0.30 - 0.65) ---
            ModelProfile(
                model_id="meta/llama-3.1-8b-instruct",
                provider="meta",
                input_cost_per_m=0.05,
                output_cost_per_m=0.08,
                capability_score=0.55,
                context_window=128_000,
                features={"tools", "json_mode", "streaming"},
                avg_latency_ms=180.0,
                description="Meta lightweight 8B model",
            ),
            ModelProfile(
                model_id="mistralai/mistral-7b-instruct",
                provider="mistral",
                input_cost_per_m=0.04,
                output_cost_per_m=0.06,
                capability_score=0.45,
                context_window=32_000,
                features={"tools", "streaming"},
                avg_latency_ms=150.0,
                description="Mistral 7B standard base worker",
            ),

            # --- RouterBench Legacy Calibration Set (for benchmark validation) ---
            ModelProfile(
                model_id="gpt-4-1106-preview",
                provider="openai",
                input_cost_per_m=10.00,
                output_cost_per_m=30.00,
                capability_score=0.90,
                context_window=128_000,
                avg_latency_ms=900.0,
            ),
            ModelProfile(
                model_id="gpt-3.5-turbo-1106",
                provider="openai",
                input_cost_per_m=1.00,
                output_cost_per_m=2.00,
                capability_score=0.68,
                context_window=16_000,
                avg_latency_ms=400.0,
            ),
            ModelProfile(
                model_id="claude-v2",
                provider="anthropic",
                input_cost_per_m=8.00,
                output_cost_per_m=24.00,
                capability_score=0.82,
                context_window=100_000,
                avg_latency_ms=800.0,
            ),
            ModelProfile(
                model_id="claude-v1",
                provider="anthropic",
                input_cost_per_m=8.00,
                output_cost_per_m=24.00,
                capability_score=0.74,
                context_window=100_000,
                avg_latency_ms=750.0,
            ),
            ModelProfile(
                model_id="claude-instant-v1",
                provider="anthropic",
                input_cost_per_m=1.63,
                output_cost_per_m=5.51,
                capability_score=0.62,
                context_window=100_000,
                avg_latency_ms=350.0,
            ),
            ModelProfile(
                model_id="mistralai/mixtral-8x7b-chat",
                provider="mistral",
                input_cost_per_m=0.60,
                output_cost_per_m=0.60,
                capability_score=0.69,
                context_window=32_000,
                avg_latency_ms=380.0,
            ),
            ModelProfile(
                model_id="mistralai/mistral-7b-chat",
                provider="mistral",
                input_cost_per_m=0.20,
                output_cost_per_m=0.20,
                capability_score=0.38,
                context_window=8_000,
                avg_latency_ms=180.0,
            ),
            ModelProfile(
                model_id="zero-one-ai/Yi-34B-Chat",
                provider="zero-one",
                input_cost_per_m=0.80,
                output_cost_per_m=0.80,
                capability_score=0.63,
                context_window=4_000,
                avg_latency_ms=450.0,
            ),
            ModelProfile(
                model_id="meta/llama-2-70b-chat",
                provider="meta",
                input_cost_per_m=0.70,
                output_cost_per_m=0.90,
                capability_score=0.60,
                context_window=4_096,
                avg_latency_ms=500.0,
            ),
            ModelProfile(
                model_id="meta/code-llama-instruct-34b-chat",
                provider="meta",
                input_cost_per_m=0.80,
                output_cost_per_m=0.80,
                capability_score=0.52,
                context_window=16_000,
                avg_latency_ms=460.0,
            ),
            ModelProfile(
                model_id="WizardLM/WizardLM-13B-V1.2",
                provider="wizardlm",
                input_cost_per_m=0.40,
                output_cost_per_m=0.40,
                capability_score=0.42,
                context_window=4_096,
                avg_latency_ms=250.0,
            ),
        ]
        for p in catalog:
            self.register(p)
