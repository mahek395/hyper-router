"""
gateway/complexity_scorer.py

Fast inference module for scoring prompt complexity in the LLM Gateway (Stage 2).
Uses the trained regression probe + structural heuristic signals.

Complexity score C is always returned in the range [0.0, 1.0].
"""

import logging
import os
import pickle
import re
from typing import Any, Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


class ComplexityScorer:
    """
    Evaluates prompt complexity C in [0.0, 1.0].

    Uses:
      1. Trained regression probe when available.
      2. Structural heuristic signals.
      3. Heuristic-only fallback if embedding/model inference fails.
    """

    def __init__(
        self,
        model_path: str = "gateway/complexity_model.pkl",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        lazy_load_embedder: bool = True,
    ):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self.lazy_load_embedder = lazy_load_embedder

        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.5

        self._embedder = None
        self._cache: Dict[str, float] = {}

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)

                self.coef = np.asarray(data["coef"], dtype=float)
                self.intercept = float(data["intercept"])

                logger.info(
                    "ComplexityScorer: loaded model artifact from '%s'",
                    self.model_path,
                )

            except Exception as exc:
                self.coef = None
                self.intercept = 0.5

                logger.warning(
                    "ComplexityScorer: failed to load model artifact '%s': %s. "
                    "Falling back to structural heuristics.",
                    self.model_path,
                    exc,
                )

        else:
            self.coef = None
            self.intercept = 0.5

            logger.warning(
                "ComplexityScorer: no model artifact at '%s' — "
                "running on structural heuristics only.",
                self.model_path,
            )

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        """
        Lazily load the sentence-transformers embedding model.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                self.embedding_model_name
            )

        return self._embedder

    # ------------------------------------------------------------------
    # Marker matching
    # ------------------------------------------------------------------

    _WORD_MARKER_RE = re.compile(r"^[a-z0-9 \-]+$")

    @classmethod
    def _matches_any(
        cls,
        lower_text: str,
        markers: list[str],
    ) -> bool:
        """
        Match ordinary words/phrases using word boundaries.

        Markers containing symbols such as:
            ```
            function(
        use substring matching instead.
        """

        for raw in markers:
            marker = raw.strip()

            if not marker:
                continue

            if cls._WORD_MARKER_RE.match(marker):
                if re.search(
                    rf"\b{re.escape(marker)}\b",
                    lower_text,
                ):
                    return True
            else:
                if marker in lower_text:
                    return True

        return False

    # ------------------------------------------------------------------
    # Structural heuristics
    # ------------------------------------------------------------------

    def extract_structural_signals(self, text: str) -> float:
        """
        Calculate heuristic complexity adjustment.

        Returns a signed boost/penalty, e.g.
            +0.22 = more complex
            -0.25 = simpler
             0.00 = neutral
        """

        boost = 0.0
        lower = text.lower()

        # --------------------------------------------------------------
        # Programming / systems complexity
        # --------------------------------------------------------------

        hard_code_markers = [
            "cuda",
            "kernel",
            "mutex",
            "deadlock",
            "concurrency",
            "websocket",
            "autodiff",
            "refactor",
            "memory leak",
            "pointer",
            "assembly",
            "bytecode",
            "optimization",
            "profiler",
            "distributed",
            "lock contention",
        ]

        if self._matches_any(lower, hard_code_markers):
            boost += 0.22

        elif self._matches_any(
            lower,
            [
                "```",
                "def",
                "class",
                "function(",
                "import",
                "select * from",
                "return",
                "const",
            ],
        ):
            boost += 0.10

        # --------------------------------------------------------------
        # Mathematics
        # --------------------------------------------------------------

        math_markers = [
            "prove that",
            "derive",
            "calculate the probability",
            "integral",
            "matrix",
            "eigenvalue",
            "theorem",
            "dirichlet",
            "multinomial",
            "bayes posterior",
            "stochastic",
            "differential equation",
            "combinatorics",
        ]

        if self._matches_any(lower, math_markers):
            boost += 0.25

        # --------------------------------------------------------------
        # Deep analysis / architecture
        # --------------------------------------------------------------

        analysis_markers = [
            "micro-architectural",
            "cache hierarchy",
            "root cause analysis",
            "tradeoffs between",
            "in-depth breakdown",
            "formal verification",
            "architecture diagram",
        ]

        if self._matches_any(lower, analysis_markers):
            boost += 0.20

        # --------------------------------------------------------------
        # Simple prompts
        # --------------------------------------------------------------

        simple_markers = [
            "what time",
            "what is the capital",
            "translate",
            "hello",
            "hi",
            "how are you",
            "synonym for",
            "who is",
            "weather in",
            "define",
            "meaning of",
        ]

        if self._matches_any(lower, simple_markers) or len(text.split()) < 10:
            if not self._matches_any(
                lower,
                [
                    "code",
                    "solve",
                    "math",
                    "why",
                    "derive",
                    "cuda",
                    "mutex",
                ],
            ):
                boost -= 0.25

        return boost

    # ------------------------------------------------------------------
    # Main scoring function
    # ------------------------------------------------------------------

    def score(
        self,
        text: str,
        embedding: Optional[Any] = None,
    ) -> float:
        """
        Return prompt complexity in the range [0.0, 1.0].

        Flow:

            prompt
              ↓
        cache lookup
              ↓
        trained model (when available)
              +
        structural heuristics
              ↓
        clamp to [0, 1]

        If the trained model or embedding process fails,
        a heuristic-only score is returned.
        """

        if not text or not text.strip():
            return 0.0

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

        cached = self._cache.get(text)

        if cached is not None:
            return cached

        # --------------------------------------------------------------
        # Structural signal
        # --------------------------------------------------------------

        structural_boost = self.extract_structural_signals(text)

        # --------------------------------------------------------------
        # Heuristic-only mode
        # --------------------------------------------------------------

        if self.coef is None:
            score = 0.5 + structural_boost
            score = max(0.0, min(1.0, score))

            self._cache[text] = score

            return score

        # --------------------------------------------------------------
        # Trained-model inference
        # --------------------------------------------------------------

        try:
            if embedding is None:
                embedding = self.embedder.encode(
                    [text],
                    normalize_embeddings=True,
                )[0]

            embedding = np.asarray(
                embedding,
                dtype=float,
            )

            # Validate dimensions.
            if embedding.ndim != 1:
                embedding = embedding.reshape(-1)

            if self.coef.shape[0] != embedding.shape[0]:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"model expects {self.coef.shape[0]}, "
                    f"received {embedding.shape[0]}"
                )

            model_score = float(
                np.dot(self.coef, embedding) + self.intercept
            )

            final_score = model_score + structural_boost

            final_score = max(
                0.0,
                min(1.0, final_score),
            )

            self._cache[text] = final_score

            return final_score

        except Exception as exc:
            logger.warning(
                "ComplexityScorer.score failed: %s — "
                "using heuristic fallback.",
                exc,
            )

            fallback_score = 0.5 + structural_boost

            fallback_score = max(
                0.0,
                min(1.0, fallback_score),
            )

            self._cache[text] = fallback_score

            return fallback_score