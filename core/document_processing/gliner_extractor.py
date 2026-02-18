"""
GLiNER-based Named Entity Recognition extractor.

Uses GLiNER2 for zero-shot NER with custom entity types.
"""

from typing import Optional

from gliner import GLiNER

from config.models import NEREntities
from config.settings import Settings


class GLiNERExtractor:
    """GLiNER-based NER extractor."""

    # Default entity types to extract
    DEFAULT_ENTITY_TYPES = [
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

    def __init__(self, settings: Settings):
        """
        Initialize GLiNER extractor.

        Args:
            settings: Application settings with model config
        """
        self.settings = settings
        self.model = GLiNER.from_pretrained(settings.gliner_model)

        # Combine default and custom entity types
        self.entity_types = self.DEFAULT_ENTITY_TYPES.copy()
        if settings.custom_entity_types:
            self.entity_types.extend(settings.custom_entity_types)

    def extract(self, text: str, threshold: float = 0.5) -> NEREntities:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from
            threshold: Confidence threshold for entity extraction

        Returns:
            NEREntities object with extracted entities
        """
        # Extract entities using GLiNER
        entities = self.model.predict_entities(
            text, self.entity_types, threshold=threshold
        )

        # Organize entities by type
        people = []
        organizations = []
        dates = []
        locations = []
        topics = []
        custom = {}
        confidence_scores = {}

        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["label"].lower()
            confidence = entity["score"]

            # Normalize entity text
            normalized_text = self._normalize_entity(entity_text)

            # Map to our schema
            if entity_type in ["person", "people"]:
                if normalized_text not in people:
                    people.append(normalized_text)
                    confidence_scores[normalized_text] = confidence

            elif entity_type in ["organization", "org", "company"]:
                if normalized_text not in organizations:
                    organizations.append(normalized_text)
                    confidence_scores[normalized_text] = confidence

            elif entity_type in ["date", "time", "datetime"]:
                if normalized_text not in dates:
                    dates.append(normalized_text)
                    confidence_scores[normalized_text] = confidence

            elif entity_type in ["location", "place", "gpe", "country", "city"]:
                if normalized_text not in locations:
                    locations.append(normalized_text)
                    confidence_scores[normalized_text] = confidence

            elif entity_type in ["topic", "subject", "theme"]:
                if normalized_text not in topics:
                    topics.append(normalized_text)
                    confidence_scores[normalized_text] = confidence

            else:
                # Custom entity type
                if entity_type not in custom:
                    custom[entity_type] = []
                if normalized_text not in custom[entity_type]:
                    custom[entity_type].append(normalized_text)
                    confidence_scores[f"{entity_type}:{normalized_text}"] = confidence

        return NEREntities(
            people=people,
            organizations=organizations,
            dates=dates,
            locations=locations,
            topics=topics,
            custom=custom,
            extractor="gliner",
            confidence_scores=confidence_scores,
        )

    def extract_batch(
        self, texts: list[str], threshold: float = 0.5
    ) -> list[NEREntities]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts to extract entities from
            threshold: Confidence threshold for entity extraction

        Returns:
            List of NEREntities objects
        """
        return [self.extract(text, threshold) for text in texts]

    def _normalize_entity(self, text: str) -> str:
        """
        Normalize entity text for deduplication.

        Args:
            text: Entity text to normalize

        Returns:
            Normalized entity text
        """
        # Strip whitespace and normalize spacing
        normalized = " ".join(text.strip().split())
        return normalized

    @property
    def extractor_name(self) -> str:
        """Return the extractor identifier."""
        return "gliner"
