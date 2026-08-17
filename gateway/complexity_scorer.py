"""
gateway/complexity_scorer.py

Fast inference module for scoring prompt complexity in the LLM Gateway (Stage 2).
Uses the trained regression probe + structural heuristic signals.
Inference time: < 0.2ms (when embedding is provided) or ~15ms (with local embedding model).
"""

import os
import pickle
import numpy as np
from typing import Optional, Dict, Any


class ComplexityScorer:
    """
    Evaluates prompt complexity C in [0.0, 1.0].
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

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.coef = data["coef"]
                self.intercept = data["intercept"]
        else:
            # Fallback uniform default if artifact not yet generated
            self.coef = None
            self.intercept = 0.5

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def extract_structural_signals(self, text: str) -> float:
        """
        Extract fast regex/structural signals to augment embedding score:
        - Code blocks, languages, or algorithmic patterns
        - Math symbols, formal proofs, and statistical formulations
        - Multi-step reasoning and deep analysis keywords
        - Conversational / simple factual query discounts
        """
        boost = 0.0
        lower = text.lower()
        
        # 1. High-difficulty code / systems programming detection
        hard_code_markers = [
            "cuda", "kernel", "mutex", "deadlock", "concurrency", "websocket",
            "autodiff", "refactor", "memory leak", "pointer", "assembly", "bytecode",
            "optimization", "profiler", "distributed", "lock contention"
        ]
        if any(m in lower for m in hard_code_markers):
            boost += 0.22
        elif any(m in lower for m in ["```", "def ", "class ", "function(", "import ", "select * from", "return ", "const "]):
            boost += 0.10

        # 2. Mathematical reasoning / proof detection
        math_markers = [
            "prove that", "derive ", "calculate the probability", "integral", "matrix",
            "eigenvalue", "theorem", "dirichlet", "multinomial", "bayes posterior",
            "stochastic", "differential equation", "combinatorics"
        ]
        if any(m in lower for m in math_markers):
            boost += 0.25

        # 3. Deep analysis & comparison indicators
        analysis_markers = [
            "micro-architectural", "cache hierarchy", "root cause analysis",
            "tradeoffs between", "in-depth breakdown", "formal verification",
            "architecture diagram"
        ]
        if any(m in lower for m in analysis_markers):
            boost += 0.20

        # 4. Simple conversational / factual / lookup discounts (easy queries)
        simple_markers = [
            "what time", "what is the capital", "translate", "hello", "hi", "how are you",
            "synonym for", "who is", "weather in", "define ", "meaning of"
        ]
        if any(m in lower for m in simple_markers) or len(text.split()) < 10:
            if not any(m in lower for m in ["code", "solve", "math", "why", "derive", "cuda", "mutex"]):
                boost -= 0.25

        return boost

    def score(
        self,
        prompt_text: str,
        embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Scores prompt complexity C in [0.0, 1.0].
        
        Args:
            prompt_text: Raw string prompt
            embedding: Optional precomputed 1D vector (dim=384)
            
        Returns:
            Continuous difficulty score from 0.0 (trivial) to 1.0 (frontier-level reasoning).
        """
        if prompt_text in self._cache:
            return self._cache[prompt_text]

        if embedding is None:
            embedding = self.embedder.encode([prompt_text], show_progress_bar=False)[0]

        if self.coef is not None:
            raw_score = float(np.dot(embedding, self.coef) + self.intercept)
        else:
            raw_score = 0.5

        # Augment with fast structural rules
        structural_adj = self.extract_structural_signals(prompt_text)
        final_score = float(np.clip(raw_score + structural_adj, 0.05, 0.99))

        # Cache query
        if len(self._cache) < 10_000:
            self._cache[prompt_text] = final_score

        return final_score

    def score_batch(
        self,
        prompts: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized batch scoring for offline evaluation."""
        if embeddings is None:
            embeddings = self.embedder.encode(prompts, show_progress_bar=False, batch_size=64)

        if self.coef is not None:
            raw_scores = np.dot(embeddings, self.coef) + self.intercept
        else:
            raw_scores = np.full(len(prompts), 0.5)

        # Apply structural adjustments vectorized
        adjustments = np.array([self.extract_structural_signals(p) for p in prompts])
        final_scores = np.clip(raw_scores + adjustments, 0.05, 0.99)
        return final_scores
