import os
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ModelFeature(str, Enum):
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"
    REASONING = "reasoning"


@dataclass
class ModelProfile:
    model_id: str
    provider: str
    base_url: str
    input_cost_per_m: float
    output_cost_per_m: float
    capability_score: float
    context_window: int = 128_000
    max_output_tokens: int = 4096
    features: Set[str] = field(default_factory=lambda: {ModelFeature.TOOLS.value, ModelFeature.JSON_MODE.value})
    avg_latency_ms: float = 600.0
    description: str = ""

    def estimate_cost(self, input_tokens: int, estimated_output_tokens: int = 250) -> float:
        cost_in = (input_tokens / 1_000_000.0) * self.input_cost_per_m
        cost_out = (estimated_output_tokens / 1_000_000.0) * self.output_cost_per_m
        return cost_in + cost_out


# Only used if Postgres is unreachable at startup, so the sidecar boots in
# degraded mode instead of refusing to start.
_FALLBACK_CATALOG = [
    ModelProfile(
        model_id="groq/llama-3.3-70b-versatile", provider="groq",
        base_url="https://api.groq.com/openai/v1",
        input_cost_per_m=0.59, output_cost_per_m=0.79, capability_score=0.60,
        features={"tools", "json_mode", "streaming"}, avg_latency_ms=800.0,
        description="Fallback catalog entry — DB was unreachable at startup",
    ),
]


class ModelRegistry:
    """
    Model pool backed by Postgres `registered_models`, written only by the
    Node admin panel. Refreshed on a timer, plus an on-demand reload Node
    triggers right after a write.
    """
    def __init__(self, refresh_interval_s: int = 20):
        self._models: Dict[str, ModelProfile] = {}
        self._lock = threading.Lock()
        self._refresh_interval_s = refresh_interval_s
        self._db_dsn = os.environ.get("DATABASE_URL")
        self.refresh()
        self._schedule_next_refresh()

    def _schedule_next_refresh(self) -> None:
        timer = threading.Timer(self._refresh_interval_s, self._background_tick)
        timer.daemon = True
        timer.start()

    def _background_tick(self) -> None:
        self.refresh()
        self._schedule_next_refresh()

    def refresh(self) -> None:
        if not self._db_dsn:
            logger.warning("ModelRegistry: DATABASE_URL not set — using fallback catalog")
            self._apply_fallback_if_empty()
            return

        try:
            import psycopg2
            import psycopg2.extras

            with psycopg2.connect(self._db_dsn) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT model_id, provider_label AS provider, base_url,
                               input_cost_per_m, output_cost_per_m, capability_score,
                               context_window, max_output_tokens, features,
                               avg_latency_ms, description
                        FROM registered_models
                    """)
                    rows = cur.fetchall()

            fresh = {
                r["model_id"]: ModelProfile(
                    model_id=r["model_id"], provider=r["provider"], base_url=r["base_url"],
                    input_cost_per_m=float(r["input_cost_per_m"]),
                    output_cost_per_m=float(r["output_cost_per_m"]),
                    capability_score=float(r["capability_score"]),
                    context_window=r["context_window"], max_output_tokens=r["max_output_tokens"],
                    features=set(r["features"] or []),
                    avg_latency_ms=float(r["avg_latency_ms"]), description=r["description"] or "",
                )
                for r in rows
            }

            with self._lock:
                self._models = fresh

            if not fresh:
                logger.warning("ModelRegistry: registered_models table is empty")

        except Exception as exc:
            logger.error("ModelRegistry: refresh failed (%s) — keeping last known pool", exc)
            self._apply_fallback_if_empty()

    def _apply_fallback_if_empty(self) -> None:
        if not self._models:
            with self._lock:
                self._models = {p.model_id: p for p in _FALLBACK_CATALOG}

    def get(self, model_id: str) -> Optional[ModelProfile]:
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> List[ModelProfile]:
        with self._lock:
            return list(self._models.values())