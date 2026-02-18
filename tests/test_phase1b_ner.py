"""
Tests for Named Entity Recognition (NER) extractors.

Tests GLiNER, spaCy, and NER Router with ensemble mode.
"""

import pytest
from unittest.mock import Mock, patch

from core.document_processing.gliner_extractor import GLiNERExtractor
from core.document_processing.spacy_extractor import SpacyExtractor
from core.document_processing.ner_router import NERRouter
from config.models import NEREntities
from config.settings import Settings


# Sample text for testing
SAMPLE_TEXT = (
    "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976. "
    "The company's revenue reached $394.3 billion in 2022."
)

SAMPLE_TEXT_2 = (
    "Microsoft Corporation, led by Satya Nadella, is headquartered in Redmond, Washington. "
    "The company announced a partnership with OpenAI in January 2023."
)


class TestGLiNERExtractor:
    """Test GLiNER-based NER extractor."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            gliner_model="urchade/gliner_small-v2.1",
            custom_entity_types=["product", "revenue"],
        )

    @pytest.fixture
    def extractor(self, settings):
        """Create GLiNER extractor instance."""
        return GLiNERExtractor(settings)

    def test_initialization(self, extractor, settings):
        """Test that GLiNER extractor initializes correctly."""
        assert extractor.settings == settings
        assert extractor.model is not None
        assert "person" in extractor.entity_types
        assert "organization" in extractor.entity_types
        assert "date" in extractor.entity_types
        assert "location" in extractor.entity_types
        assert "topic" in extractor.entity_types

    def test_default_entity_types(self, extractor):
        """Test that default entity types are included."""
        expected_types = [
            "person",
            "organization",
            "date",
            "location",
            "topic",
            "event",
            "product",
            "money",
            "percentage",
        ]
        for entity_type in expected_types:
            assert entity_type in extractor.entity_types

    def test_custom_entity_types(self, extractor):
        """Test that custom entity types are added."""
        assert "product" in extractor.entity_types
        assert "revenue" in extractor.entity_types

    def test_extract_entities(self, extractor):
        """Test entity extraction with default entity types."""
        entities = extractor.extract(SAMPLE_TEXT, threshold=0.3)

        # Verify return type
        assert isinstance(entities, NEREntities)
        assert entities.extractor == "gliner"

        # Verify entity fields exist
        assert isinstance(entities.people, list)
        assert isinstance(entities.organizations, list)
        assert isinstance(entities.dates, list)
        assert isinstance(entities.locations, list)
        assert isinstance(entities.topics, list)
        assert isinstance(entities.custom, dict)
        assert isinstance(entities.confidence_scores, dict)

        # Verify some expected entities are found
        # Note: Exact entities may vary based on model, so we check for presence
        assert len(entities.people) > 0 or len(entities.organizations) > 0

    def test_confidence_scores(self, extractor):
        """Test that confidence scores are provided for entities."""
        entities = extractor.extract(SAMPLE_TEXT, threshold=0.3)

        # Check that confidence scores exist
        assert len(entities.confidence_scores) > 0

        # Verify all scores are between 0 and 1
        for entity, score in entities.confidence_scores.items():
            assert 0.0 <= score <= 1.0

    def test_entity_deduplication(self, extractor):
        """Test that duplicate entities are removed."""
        # Text with duplicate entities
        text = "Steve Jobs and Steve Jobs founded Apple. Apple Inc. is a company."
        entities = extractor.extract(text, threshold=0.3)

        # Check that each entity appears only once in its category
        assert len(entities.people) == len(set(entities.people))
        assert len(entities.organizations) == len(set(entities.organizations))
        assert len(entities.locations) == len(set(entities.locations))

    def test_threshold_filtering(self, extractor):
        """Test that threshold filters low-confidence entities."""
        # Extract with high threshold
        high_threshold_entities = extractor.extract(SAMPLE_TEXT, threshold=0.8)

        # Extract with low threshold
        low_threshold_entities = extractor.extract(SAMPLE_TEXT, threshold=0.3)

        # Low threshold should find same or more entities
        total_high = (
            len(high_threshold_entities.people)
            + len(high_threshold_entities.organizations)
            + len(high_threshold_entities.dates)
            + len(high_threshold_entities.locations)
        )
        total_low = (
            len(low_threshold_entities.people)
            + len(low_threshold_entities.organizations)
            + len(low_threshold_entities.dates)
            + len(low_threshold_entities.locations)
        )

        assert total_low >= total_high

    def test_batch_processing(self, extractor):
        """Test batch processing of multiple texts."""
        texts = [SAMPLE_TEXT, SAMPLE_TEXT_2]
        results = extractor.extract_batch(texts, threshold=0.3)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, NEREntities) for r in results)
        assert all(r.extractor == "gliner" for r in results)

    def test_normalize_entity(self, extractor):
        """Test entity text normalization."""
        # Test with extra whitespace
        normalized = extractor._normalize_entity("  Steve   Jobs  ")
        assert normalized == "Steve Jobs"

        # Test with newlines
        normalized = extractor._normalize_entity("Apple\nInc.")
        assert normalized == "Apple Inc."

    def test_extractor_name(self, extractor):
        """Test extractor name property."""
        assert extractor.extractor_name == "gliner"

    def test_empty_text(self, extractor):
        """Test extraction from empty text."""
        entities = extractor.extract("", threshold=0.5)

        assert isinstance(entities, NEREntities)
        assert len(entities.people) == 0
        assert len(entities.organizations) == 0
        assert len(entities.dates) == 0
        assert len(entities.locations) == 0


class TestSpacyExtractor:
    """Test spaCy-based NER extractor."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(spacy_model="en_core_web_sm")

    @pytest.fixture
    def extractor(self, settings):
        """Create spaCy extractor instance."""
        try:
            return SpacyExtractor(settings)
        except ValueError as e:
            pytest.skip(f"spaCy model not available: {e}")

    def test_initialization(self, extractor, settings):
        """Test that spaCy extractor initializes correctly."""
        assert extractor.settings == settings
        assert extractor.nlp is not None

    def test_label_mapping(self, extractor):
        """Test that label mapping is defined correctly."""
        assert "PERSON" in extractor.LABEL_MAPPING
        assert "ORG" in extractor.LABEL_MAPPING
        assert "DATE" in extractor.LABEL_MAPPING
        assert "GPE" in extractor.LABEL_MAPPING
        assert "LOC" in extractor.LABEL_MAPPING

        # Verify mappings
        assert extractor.LABEL_MAPPING["PERSON"] == "people"
        assert extractor.LABEL_MAPPING["ORG"] == "organizations"
        assert extractor.LABEL_MAPPING["DATE"] == "dates"
        assert extractor.LABEL_MAPPING["GPE"] == "locations"
        assert extractor.LABEL_MAPPING["LOC"] == "locations"

    def test_extract_standard_entities(self, extractor):
        """Test extraction of standard NER entities."""
        entities = extractor.extract(SAMPLE_TEXT)

        # Verify return type
        assert isinstance(entities, NEREntities)
        assert entities.extractor == "spacy"

        # Verify entity fields exist
        assert isinstance(entities.people, list)
        assert isinstance(entities.organizations, list)
        assert isinstance(entities.dates, list)
        assert isinstance(entities.locations, list)
        assert isinstance(entities.topics, list)
        assert isinstance(entities.custom, dict)
        assert isinstance(entities.confidence_scores, dict)

    def test_noun_chunk_extraction(self, extractor):
        """Test noun chunk extraction as topics."""
        entities = extractor.extract(SAMPLE_TEXT, extract_noun_chunks=True)

        # Topics should be extracted from noun chunks
        assert isinstance(entities.topics, list)

        # Verify topics are multi-word phrases (2-5 words)
        for topic in entities.topics:
            word_count = len(topic.split())
            assert 2 <= word_count <= 5

    def test_noun_chunk_disabled(self, extractor):
        """Test that noun chunk extraction can be disabled."""
        entities = extractor.extract(SAMPLE_TEXT, extract_noun_chunks=False)

        # Topics should be empty when disabled
        assert len(entities.topics) == 0

    def test_confidence_scores(self, extractor):
        """Test that confidence scores are provided."""
        entities = extractor.extract(SAMPLE_TEXT)

        # spaCy uses fixed confidence of 0.9 for entities
        for entity in entities.people + entities.organizations + entities.dates + entities.locations:
            if entity in entities.confidence_scores:
                assert entities.confidence_scores[entity] == 0.9

        # Topics have confidence of 0.7
        for topic in entities.topics:
            topic_key = f"topic:{topic}"
            if topic_key in entities.confidence_scores:
                assert entities.confidence_scores[topic_key] == 0.7

    def test_entity_deduplication(self, extractor):
        """Test that duplicate entities are removed."""
        text = "Apple Inc. and Apple Inc. are the same company. Apple is in California."
        entities = extractor.extract(text)

        # Check that each entity appears only once
        assert len(entities.organizations) == len(set(entities.organizations))
        assert len(entities.locations) == len(set(entities.locations))

    def test_batch_processing(self, extractor):
        """Test batch processing with spacy.pipe()."""
        texts = [SAMPLE_TEXT, SAMPLE_TEXT_2]
        results = extractor.extract_batch(texts)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, NEREntities) for r in results)
        assert all(r.extractor == "spacy" for r in results)

    def test_extract_from_doc(self, extractor):
        """Test extraction from spaCy Doc object."""
        doc = extractor.nlp(SAMPLE_TEXT)
        entities = extractor._extract_from_doc(doc, extract_noun_chunks=True)

        assert isinstance(entities, NEREntities)
        assert entities.extractor == "spacy"

    def test_normalize_entity(self, extractor):
        """Test entity text normalization."""
        normalized = extractor._normalize_entity("  Apple   Inc.  ")
        assert normalized == "Apple Inc."

    def test_extractor_name(self, extractor):
        """Test extractor name property."""
        assert extractor.extractor_name == "spacy"

    def test_topic_limit(self, extractor):
        """Test that topics are limited to top 10."""
        # Create text with many noun chunks
        long_text = " ".join([f"The big company number {i}" for i in range(20)])
        entities = extractor.extract(long_text, extract_noun_chunks=True)

        # Should be limited to 10 topics
        assert len(entities.topics) <= 10

    def test_empty_text(self, extractor):
        """Test extraction from empty text."""
        entities = extractor.extract("")

        assert isinstance(entities, NEREntities)
        assert len(entities.people) == 0
        assert len(entities.organizations) == 0
        assert len(entities.dates) == 0
        assert len(entities.locations) == 0

    def test_invalid_model(self):
        """Test that invalid model raises ValueError."""
        settings = Settings(spacy_model="invalid_model_name")
        with pytest.raises(ValueError, match="spaCy model .* not found"):
            SpacyExtractor(settings)


class TestNERRouter:
    """Test NER router with ensemble mode."""

    @pytest.fixture
    def settings_gliner(self):
        """Create settings for GLiNER mode."""
        return Settings(
            ner_mode="gliner",
            gliner_model="urchade/gliner_small-v2.1",
        )

    @pytest.fixture
    def settings_spacy(self):
        """Create settings for spaCy mode."""
        return Settings(
            ner_mode="spacy",
            spacy_model="en_core_web_sm",
        )

    @pytest.fixture
    def settings_ensemble(self):
        """Create settings for ensemble mode."""
        return Settings(
            ner_mode="ensemble",
            gliner_model="urchade/gliner_small-v2.1",
            spacy_model="en_core_web_sm",
        )

    def test_initialization_gliner_mode(self, settings_gliner):
        """Test router initialization in GLiNER mode."""
        router = NERRouter(settings_gliner)

        assert router.mode == "gliner"
        assert router.gliner_extractor is not None
        assert router.spacy_extractor is None

    def test_initialization_spacy_mode(self, settings_spacy):
        """Test router initialization in spaCy mode."""
        try:
            router = NERRouter(settings_spacy)
            assert router.mode == "spacy"
            assert router.gliner_extractor is None
            assert router.spacy_extractor is not None
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_initialization_ensemble_mode(self, settings_ensemble):
        """Test router initialization in ensemble mode."""
        try:
            router = NERRouter(settings_ensemble)
            assert router.mode == "ensemble"
            assert router.gliner_extractor is not None
            assert router.spacy_extractor is not None
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_extract_gliner_mode(self, settings_gliner):
        """Test extraction in GLiNER mode."""
        router = NERRouter(settings_gliner)
        entities = router.extract(SAMPLE_TEXT)

        assert isinstance(entities, NEREntities)
        assert entities.extractor == "gliner"

    def test_extract_spacy_mode(self, settings_spacy):
        """Test extraction in spaCy mode."""
        try:
            router = NERRouter(settings_spacy)
            entities = router.extract(SAMPLE_TEXT)

            assert isinstance(entities, NEREntities)
            assert entities.extractor == "spacy"
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_extract_ensemble_mode(self, settings_ensemble):
        """Test extraction in ensemble mode."""
        try:
            router = NERRouter(settings_ensemble)
            entities = router.extract(SAMPLE_TEXT)

            assert isinstance(entities, NEREntities)
            assert entities.extractor == "ensemble"
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_ensemble_merges_results(self, settings_ensemble):
        """Test that ensemble mode merges results from both extractors."""
        try:
            router = NERRouter(settings_ensemble)
            entities = router.extract(SAMPLE_TEXT)

            # Should have entities from both extractors
            assert isinstance(entities, NEREntities)
            assert entities.extractor == "ensemble"

            # Verify no duplicates
            assert len(entities.people) == len(set(entities.people))
            assert len(entities.organizations) == len(set(entities.organizations))
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_confidence_boosting(self, settings_ensemble):
        """Test that entities found by both extractors get boosted confidence."""
        try:
            router = NERRouter(settings_ensemble)

            # Get individual results
            gliner_entities = router.gliner_extractor.extract(SAMPLE_TEXT)
            spacy_entities = router.spacy_extractor.extract(SAMPLE_TEXT)

            # Get ensemble results
            ensemble_entities = router.extract(SAMPLE_TEXT)

            # Find entities present in both
            common_people = set(gliner_entities.people) & set(spacy_entities.people)
            common_orgs = set(gliner_entities.organizations) & set(spacy_entities.organizations)

            # Check that common entities have boosted confidence
            for entity in common_people | common_orgs:
                if entity in ensemble_entities.confidence_scores:
                    # Ensemble confidence should be higher than individual
                    gliner_score = gliner_entities.confidence_scores.get(entity, 0.0)
                    spacy_score = spacy_entities.confidence_scores.get(entity, 0.0)
                    ensemble_score = ensemble_entities.confidence_scores[entity]

                    # Should be boosted (average + 0.1, capped at 0.95)
                    if gliner_score > 0 and spacy_score > 0:
                        expected_boost = min(0.95, (gliner_score + spacy_score) / 2 + 0.1)
                        assert ensemble_score >= max(gliner_score, spacy_score)
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_batch_processing_gliner(self, settings_gliner):
        """Test batch processing in GLiNER mode."""
        router = NERRouter(settings_gliner)
        texts = [SAMPLE_TEXT, SAMPLE_TEXT_2]
        results = router.extract_batch(texts)

        assert len(results) == 2
        assert all(isinstance(r, NEREntities) for r in results)
        assert all(r.extractor == "gliner" for r in results)

    def test_batch_processing_spacy(self, settings_spacy):
        """Test batch processing in spaCy mode."""
        try:
            router = NERRouter(settings_spacy)
            texts = [SAMPLE_TEXT, SAMPLE_TEXT_2]
            results = router.extract_batch(texts)

            assert len(results) == 2
            assert all(isinstance(r, NEREntities) for r in results)
            assert all(r.extractor == "spacy" for r in results)
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_batch_processing_ensemble(self, settings_ensemble):
        """Test batch processing in ensemble mode."""
        try:
            router = NERRouter(settings_ensemble)
            texts = [SAMPLE_TEXT, SAMPLE_TEXT_2]
            results = router.extract_batch(texts)

            assert len(results) == 2
            assert all(isinstance(r, NEREntities) for r in results)
            assert all(r.extractor == "ensemble" for r in results)
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_merge_entities(self, settings_ensemble):
        """Test entity merging logic."""
        try:
            router = NERRouter(settings_ensemble)

            # Create mock entities
            gliner_entities = NEREntities(
                people=["Steve Jobs", "Tim Cook"],
                organizations=["Apple Inc."],
                dates=["1976"],
                locations=["Cupertino"],
                topics=["technology"],
                custom={"product": ["iPhone"]},
                extractor="gliner",
                confidence_scores={
                    "Steve Jobs": 0.85,
                    "Apple Inc.": 0.90,
                    "1976": 0.75,
                },
            )

            spacy_entities = NEREntities(
                people=["Steve Jobs"],  # Duplicate
                organizations=["Apple Inc.", "Microsoft"],  # One duplicate, one new
                dates=["April 1, 1976"],
                locations=["California"],
                topics=["innovation"],
                custom={"money": ["$394.3 billion"]},
                extractor="spacy",
                confidence_scores={
                    "Steve Jobs": 0.9,
                    "Apple Inc.": 0.9,
                    "April 1, 1976": 0.9,
                },
            )

            merged = router._merge_entities(gliner_entities, spacy_entities)

            # Verify merging
            assert "Steve Jobs" in merged.people
            assert "Tim Cook" in merged.people
            assert "Apple Inc." in merged.organizations
            assert "Microsoft" in merged.organizations

            # Verify confidence boosting for duplicates
            assert merged.confidence_scores["Steve Jobs"] > 0.85
            assert merged.confidence_scores["Apple Inc."] > 0.90

            # Verify custom entities merged
            assert "product" in merged.custom
            assert "money" in merged.custom
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_merge_list(self, settings_ensemble):
        """Test list merging with deduplication."""
        try:
            router = NERRouter(settings_ensemble)

            list1 = ["Apple Inc.", "Microsoft"]
            list2 = ["apple inc.", "Google"]  # Case-insensitive duplicate

            merged = router._merge_list(list1, list2)

            # Should have 3 items (Apple Inc. deduplicated)
            assert len(merged) == 3
            assert "Apple Inc." in merged or "apple inc." in merged
            assert "Microsoft" in merged
            assert "Google" in merged
        except ValueError:
            pytest.skip("spaCy model not available")

    def test_extractor_name(self, settings_gliner):
        """Test extractor name property."""
        router = NERRouter(settings_gliner)
        assert router.extractor_name == "gliner"

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        settings = Settings(ner_mode="invalid_mode")
        router = NERRouter(settings)

        with pytest.raises(ValueError, match="Unsupported NER mode"):
            router.extract(SAMPLE_TEXT)

    def test_invalid_mode_batch(self):
        """Test that invalid mode raises ValueError in batch processing."""
        settings = Settings(ner_mode="invalid_mode")
        router = NERRouter(settings)

        with pytest.raises(ValueError, match="Unsupported NER mode"):
            router.extract_batch([SAMPLE_TEXT])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
