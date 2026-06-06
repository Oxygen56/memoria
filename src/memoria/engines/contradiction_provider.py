"""Contradiction detection providers — LLM and local model implementations.

Supports configurable providers:
- LLMContradictionProvider: Uses deepseek-v4-pro (default) via OpenAI-compatible API
- CrossEncoderContradictionProvider: Local sentence-transformers model
- HeuristicContradictionProvider: Fast regex-based fallback (no external deps)
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Stop words (shared with feedback.py heuristic) ─────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same",
    "it", "its", "this", "that", "these", "those", "he",
    "she", "they", "them", "their", "his", "her", "my",
    "your", "our", "we", "you", "i", "me", "us",
}

_NEGATION_MARKERS = ("not ", "never ", "don't ", "doesn't ", "isn't ", "aren't ", "won't ", "can't ")

_SYSTEM_PROMPT = (
    "You are a contradiction detector. Given two statements about a system or "
    "project, determine if they contradict each other. Respond ONLY with JSON: "
    '{"is_contradiction": bool, "confidence": 0.0-1.0, "explanation": "..."}'
)


# ── Result dataclass ──────────────────────────────


@dataclass
class ContradictionResult:
    """Result of a contradiction check."""

    is_contradiction: bool = False
    confidence: float = 0.0
    explanation: str = ""
    method: str = ""  # "llm", "cross_encoder", "heuristic"


# ── Abstract base ─────────────────────────────────


class ContradictionProvider(ABC):
    """Abstract base for contradiction detection."""

    @abstractmethod
    async def check(self, content_a: str, content_b: str) -> ContradictionResult:
        """Check if two memory contents contradict each other."""
        ...


# ── LLM Provider ──────────────────────────────────


class LLMContradictionProvider(ContradictionProvider):
    """Uses an LLM (default: deepseek-v4-pro) for contradiction detection.

    Uses OpenAI-compatible API format via httpx for flexibility and minimal deps.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_base = (
            api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        ).rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    async def check(self, content_a: str, content_b: str) -> ContradictionResult:
        """Use LLM to classify contradiction via OpenAI-compatible chat API."""
        import httpx

        url = f"{self._api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Statement A: {content_a}\n\nStatement B: {content_b}\n\n"
                        "Are these two statements contradictory?"
                    ),
                },
            ],
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]

        # Parse JSON from LLM response (strip markdown fences if present)
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM contradiction response: %s", raw_content)
            return ContradictionResult(
                is_contradiction=False,
                confidence=0.0,
                explanation="Failed to parse LLM response",
                method="llm",
            )

        return ContradictionResult(
            is_contradiction=bool(result.get("is_contradiction", False)),
            confidence=float(result.get("confidence", 0.0)),
            explanation=str(result.get("explanation", "")),
            method="llm",
        )


# ── CrossEncoder Provider ─────────────────────────


class CrossEncoderContradictionProvider(ContradictionProvider):
    """Uses a local CrossEncoder model for NLI-based contradiction detection.

    Model: cross-encoder/nli-deberta-v3-base (supports: contradiction, entailment, neutral)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        contradiction_threshold: float = 0.7,
    ):
        self._model_name = model_name
        self._contradiction_threshold = contradiction_threshold
        self._model = None  # Lazy load

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for CrossEncoder provider. "
                "Install: pip install sentence-transformers"
            )

    async def check(self, content_a: str, content_b: str) -> ContradictionResult:
        """Use CrossEncoder NLI to detect contradiction.

        Output label order: [contradiction, entailment, neutral]
        """
        self._load_model()

        # CrossEncoder predict returns scores for each label
        scores = self._model.predict(  # type: ignore[union-attr]
            [(content_a, content_b)],
            apply_softmax=True,
        )

        # scores shape: (1, 3) → [contradiction, entailment, neutral]
        probs = scores[0]
        contradiction_prob = float(probs[0])
        is_contradiction = contradiction_prob > self._contradiction_threshold

        return ContradictionResult(
            is_contradiction=is_contradiction,
            confidence=contradiction_prob,
            explanation=(
                f"NLI scores — contradiction: {probs[0]:.3f}, "
                f"entailment: {probs[1]:.3f}, neutral: {probs[2]:.3f}"
            ),
            method="cross_encoder",
        )


# ── Heuristic Provider ────────────────────────────


class HeuristicContradictionProvider(ContradictionProvider):
    """Fast, no-cost heuristic contradiction detection (fallback).

    Uses negation pattern matching and numeric contradiction detection.
    No external dependencies required.
    """

    def __init__(self, overlap_threshold: int = 3):
        self._overlap_threshold = overlap_threshold

    async def check(self, content_a: str, content_b: str) -> ContradictionResult:
        """Use regex patterns for basic contradiction detection."""
        text_a = content_a.lower()
        text_b = content_b.lower()

        # 1. Negation pattern check
        negation_a = any(marker in text_a for marker in _NEGATION_MARKERS)
        negation_b = any(marker in text_b for marker in _NEGATION_MARKERS)

        if negation_a != negation_b:
            terms_a = set(text_a.split()) - _STOP_WORDS
            terms_b = set(text_b.split()) - _STOP_WORDS
            overlap = terms_a & terms_b
            if len(overlap) >= self._overlap_threshold:
                return ContradictionResult(
                    is_contradiction=True,
                    confidence=0.6,
                    explanation=f"Negation pattern with {len(overlap)} shared terms: {overlap}",
                    method="heuristic",
                )

        # 2. Numeric contradiction: same topic, different values
        numbers_a = re.findall(r"\b(\d+)\b", text_a)
        numbers_b = re.findall(r"\b(\d+)\b", text_b)
        if numbers_a and numbers_b and numbers_a != numbers_b:
            terms_a = set(text_a.split()) - _STOP_WORDS
            terms_b = set(text_b.split()) - _STOP_WORDS
            if len(terms_a & terms_b) >= self._overlap_threshold:
                return ContradictionResult(
                    is_contradiction=True,
                    confidence=0.5,
                    explanation=(
                        f"Numeric mismatch: {numbers_a} vs {numbers_b} "
                        f"with shared context terms"
                    ),
                    method="heuristic",
                )

        return ContradictionResult(
            is_contradiction=False,
            confidence=0.1,
            explanation="No contradiction patterns detected",
            method="heuristic",
        )


# ── Fallback Wrapper ──────────────────────────────


class FallbackContradictionProvider(ContradictionProvider):
    """Wraps a primary provider with fallback on failure."""

    def __init__(self, primary: ContradictionProvider, fallback: ContradictionProvider):
        self._primary = primary
        self._fallback = fallback

    async def check(self, content_a: str, content_b: str) -> ContradictionResult:
        """Try primary provider; on any exception, use fallback."""
        try:
            return await self._primary.check(content_a, content_b)
        except Exception as e:
            logger.warning(
                "Primary contradiction provider failed: %s, using fallback", e
            )
            return await self._fallback.check(content_a, content_b)


# ── Factory ───────────────────────────────────────


class ContradictionProviderFactory:
    """Factory to create contradiction providers based on config."""

    @staticmethod
    def create(
        provider_type: str = "llm",
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        fallback_type: str = "heuristic",
        **kwargs,
    ) -> ContradictionProvider:
        """Create a provider with optional fallback wrapping.

        Args:
            provider_type: Primary provider type — "llm", "cross_encoder", or "heuristic".
            model: Model name for LLM provider (default: deepseek-v4-pro).
            api_key: API key for LLM provider (falls back to OPENAI_API_KEY env var).
            api_base: API base URL (falls back to OPENAI_API_BASE env var).
            fallback_type: Fallback provider type — "heuristic" or "cross_encoder".
            **kwargs: Additional keyword arguments passed to the provider constructor.

        Returns:
            A ContradictionProvider, potentially wrapped with FallbackContradictionProvider.
        """
        primary = ContradictionProviderFactory._build_provider(
            provider_type, model=model, api_key=api_key, api_base=api_base, **kwargs
        )

        # If primary is already heuristic, no fallback needed
        if provider_type == "heuristic":
            return primary

        fallback = ContradictionProviderFactory._build_provider(fallback_type)
        return FallbackContradictionProvider(primary=primary, fallback=fallback)

    @staticmethod
    def _build_provider(
        provider_type: str,
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> ContradictionProvider:
        """Instantiate a single provider by type string."""
        if provider_type == "llm":
            return LLMContradictionProvider(
                model=model,
                api_key=api_key,
                api_base=api_base,
                **kwargs,
            )
        elif provider_type == "cross_encoder":
            return CrossEncoderContradictionProvider(**kwargs)
        elif provider_type == "heuristic":
            return HeuristicContradictionProvider(**kwargs)
        else:
            raise ValueError(
                f"Unknown contradiction provider type: '{provider_type}'. "
                f"Available: llm, cross_encoder, heuristic"
            )
