"""Tests for contradiction detection providers."""

import pytest

from memoria.engines.contradiction_provider import (
    ContradictionProvider,
    ContradictionProviderFactory,
    ContradictionResult,
    FallbackContradictionProvider,
    HeuristicContradictionProvider,
)


@pytest.fixture
def heuristic_provider():
    """Create a HeuristicContradictionProvider instance."""
    return HeuristicContradictionProvider()


class TestHeuristicProvider:
    """Test the heuristic contradiction provider."""

    async def test_negation_contradiction(self, heuristic_provider):
        """Should detect negation pattern with shared terms."""
        result = await heuristic_provider.check(
            "Redis pool size is 10",
            "Redis pool size is not 10",
        )
        assert isinstance(result, ContradictionResult)
        assert result.method == "heuristic"
        # Should detect: one has negation "not", shared terms "redis", "pool", "size", "10"
        assert result.is_contradiction is True

    async def test_numeric_contradiction(self, heuristic_provider):
        """Should detect numeric contradictions — same topic, different numbers."""
        result = await heuristic_provider.check(
            "Redis pool size is 10",
            "Redis pool size is 50",
        )
        assert isinstance(result, ContradictionResult)
        assert result.is_contradiction is True
        assert result.confidence > 0

    async def test_no_false_positive_complementary(self, heuristic_provider):
        """Non-contradictory complementary statements should not trigger."""
        result = await heuristic_provider.check(
            "We use PostgreSQL",
            "PostgreSQL version is 16",
        )
        assert result.is_contradiction is False

    async def test_unrelated_content(self, heuristic_provider):
        """Completely unrelated content should not be flagged."""
        result = await heuristic_provider.check(
            "The sky is blue",
            "Docker uses containers",
        )
        assert result.is_contradiction is False

    async def test_confidence_range(self, heuristic_provider):
        """Confidence should be between 0 and 1."""
        result = await heuristic_provider.check(
            "Redis port is 6379",
            "Redis port is 6380",
        )
        assert 0.0 <= result.confidence <= 1.0

    async def test_empty_strings(self, heuristic_provider):
        """Empty strings should not crash and should return no contradiction."""
        result = await heuristic_provider.check("", "")
        assert result.is_contradiction is False

    async def test_same_content(self, heuristic_provider):
        """Identical content should not be flagged as contradiction."""
        result = await heuristic_provider.check(
            "Redis pool size is 10",
            "Redis pool size is 10",
        )
        assert result.is_contradiction is False

    async def test_negation_without_shared_context(self, heuristic_provider):
        """Negation without enough shared terms should not trigger."""
        result = await heuristic_provider.check(
            "I don't like apples",
            "Bananas are great for health",
        )
        assert result.is_contradiction is False


class TestFallbackProvider:
    """Test the fallback mechanism."""

    async def test_fallback_on_error(self):
        """Should fall back to secondary provider when primary fails."""

        class FailingProvider(ContradictionProvider):
            async def check(self, content_a: str, content_b: str) -> ContradictionResult:
                raise RuntimeError("API unavailable")

        heuristic = HeuristicContradictionProvider()
        fallback = FallbackContradictionProvider(FailingProvider(), heuristic)

        result = await fallback.check("test a", "test b")
        assert isinstance(result, ContradictionResult)
        assert result.method == "heuristic"

    async def test_primary_success_no_fallback(self):
        """When primary succeeds, fallback should not be called."""

        class SuccessProvider(ContradictionProvider):
            async def check(self, content_a: str, content_b: str) -> ContradictionResult:
                return ContradictionResult(
                    is_contradiction=True,
                    confidence=0.99,
                    explanation="Test",
                    method="custom",
                )

        class NeverCalledProvider(ContradictionProvider):
            async def check(self, content_a: str, content_b: str) -> ContradictionResult:
                raise AssertionError("Fallback should not be called")

        provider = FallbackContradictionProvider(SuccessProvider(), NeverCalledProvider())
        result = await provider.check("a", "b")
        assert result.is_contradiction is True
        assert result.method == "custom"


class TestFactory:
    """Test provider factory."""

    def test_create_heuristic(self):
        """Factory should create a heuristic provider without wrapping."""
        provider = ContradictionProviderFactory.create(provider_type="heuristic")
        assert provider is not None
        assert isinstance(provider, HeuristicContradictionProvider)

    def test_create_llm_with_fallback(self):
        """Factory should wrap LLM provider with fallback."""
        provider = ContradictionProviderFactory.create(
            provider_type="llm",
            api_key="test-key",
        )
        assert provider is not None
        # LLM type gets wrapped in FallbackContradictionProvider
        assert isinstance(provider, FallbackContradictionProvider)

    def test_create_unknown_raises(self):
        """Factory should raise ValueError for unknown provider type."""
        with pytest.raises(ValueError, match="Unknown contradiction provider type"):
            ContradictionProviderFactory.create(provider_type="nonexistent")

    async def test_heuristic_provider_from_factory_works(self):
        """Heuristic provider from factory should produce valid results."""
        provider = ContradictionProviderFactory.create(provider_type="heuristic")
        result = await provider.check(
            "Server runs on port 8080",
            "Server runs on port 9090",
        )
        assert isinstance(result, ContradictionResult)
        assert result.is_contradiction is True
