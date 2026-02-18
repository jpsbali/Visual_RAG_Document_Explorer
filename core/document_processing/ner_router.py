"""
NER router for ensemble extraction.

Routes NER requests to GLiNER, spaCy, or both (ensemble mode).
"""

from typing import Literal

from config.models import NEREntities
from config.settings import Settings
from core.document_processing.gliner_extractor import GLiNERExtractor
from core.document_processing.spacy_extractor import SpacyExtractor


class NERRouter:
    """Router for NER extraction with ensemble support."""

    def __init__(self, settings: Settings):
        """
        Initialize NER router.

        Args:
            settings: Application settings with NER mode config
        """
        self.settings = settings
        self.mode = settings.ner_mode

        # Initialize extractors based on mode
        self.gliner_extractor = None
        self.spacy_extractor = None

        if self.mode in ["gliner", "ensemble"]:
            self.gliner_extractor = GLiNERExtractor(settings)

        if self.mode in ["spacy", "ensemble"]:
            self.spacy_extractor = SpacyExtractor(settings)

    def extract(self, text: str) -> NEREntities:
        """
        Extract named entities from text using configured mode.

        Args:
            text: Text to extract entities from

        Returns:
            NEREntities object with extracted entities
        """
        if self.mode == "gliner":
            return self.gliner_extractor.extract(text)

        elif self.mode == "spacy":
            return self.spacy_extractor.extract(text)

        elif self.mode == "ensemble":
            # Run both extractors and merge results
            gliner_entities = self.gliner_extractor.extract(text)
            spacy_entities = self.spacy_extractor.extract(text)
            return self._merge_entities(gliner_entities, spacy_entities)

        else:
            raise ValueError(
                f"Unsupported NER mode: {self.mode}. "
                f"Supported modes: gliner, spacy, ensemble"
            )

    def extract_batch(self, texts: list[str]) -> list[NEREntities]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts to extract entities from

        Returns:
            List of NEREntities objects
        """
        if self.mode == "gliner":
            return self.gliner_extractor.extract_batch(texts)

        elif self.mode == "spacy":
            return self.spacy_extractor.extract_batch(texts)

        elif self.mode == "ensemble":
            # Run both extractors and merge results for each text
            gliner_results = self.gliner_extractor.extract_batch(texts)
            spacy_results = self.spacy_extractor.extract_batch(texts)
            return [
                self._merge_entities(gliner, spacy)
                for gliner, spacy in zip(gliner_results, spacy_results)
            ]

        else:
            raise ValueError(
                f"Unsupported NER mode: {self.mode}. "
                f"Supported modes: gliner, spacy, ensemble"
            )

    def _merge_entities(
        self, gliner_entities: NEREntities, spacy_entities: NEREntities
    ) -> NEREntities:
        """
        Merge entities from GLiNER and spaCy extractors.

        Entities found by both extractors get higher confidence scores.
        Deduplicates entities across extractors.

        Args:
            gliner_entities: Entities from GLiNER
            spacy_entities: Entities from spaCy

        Returns:
            Merged NEREntities object
        """
        # Merge people
        people = self._merge_list(
            gliner_entities.people,
            spacy_entities.people,
        )

        # Merge organizations
        organizations = self._merge_list(
            gliner_entities.organizations,
            spacy_entities.organizations,
        )

        # Merge dates
        dates = self._merge_list(
            gliner_entities.dates,
            spacy_entities.dates,
        )

        # Merge locations
        locations = self._merge_list(
            gliner_entities.locations,
            spacy_entities.locations,
        )

        # Merge topics
        topics = self._merge_list(
            gliner_entities.topics,
            spacy_entities.topics,
        )

        # Merge custom entities
        custom = {}
        for key, values in gliner_entities.custom.items():
            custom[key] = values.copy()

        for key, values in spacy_entities.custom.items():
            if key in custom:
                custom[key] = self._merge_list(custom[key], values)
            else:
                custom[key] = values.copy()

        # Merge confidence scores
        confidence_scores = {}

        # Process all entity lists
        all_entities = {
            "people": people,
            "organizations": organizations,
            "dates": dates,
            "locations": locations,
            "topics": topics,
        }

        for category, entity_list in all_entities.items():
            for entity in entity_list:
                # Check if entity is in both extractors
                gliner_score = gliner_entities.confidence_scores.get(entity, 0.0)
                spacy_score = spacy_entities.confidence_scores.get(entity, 0.0)

                if gliner_score > 0 and spacy_score > 0:
                    # Found by both - boost confidence
                    confidence_scores[entity] = min(0.95, (gliner_score + spacy_score) / 2 + 0.1)
                elif gliner_score > 0:
                    confidence_scores[entity] = gliner_score
                elif spacy_score > 0:
                    confidence_scores[entity] = spacy_score

        # Process custom entities
        for key, values in custom.items():
            for entity in values:
                gliner_key = f"{key}:{entity}"
                spacy_key = f"{key}:{entity}"

                gliner_score = gliner_entities.confidence_scores.get(gliner_key, 0.0)
                spacy_score = spacy_entities.confidence_scores.get(spacy_key, 0.0)

                if gliner_score > 0 and spacy_score > 0:
                    confidence_scores[gliner_key] = min(0.95, (gliner_score + spacy_score) / 2 + 0.1)
                elif gliner_score > 0:
                    confidence_scores[gliner_key] = gliner_score
                elif spacy_score > 0:
                    confidence_scores[spacy_key] = spacy_score

        return NEREntities(
            people=people,
            organizations=organizations,
            dates=dates,
            locations=locations,
            topics=topics,
            custom=custom,
            extractor="ensemble",
            confidence_scores=confidence_scores,
        )

    def _merge_list(self, list1: list[str], list2: list[str]) -> list[str]:
        """
        Merge two lists, removing duplicates while preserving order.

        Args:
            list1: First list
            list2: Second list

        Returns:
            Merged list without duplicates
        """
        # Use dict to preserve order while removing duplicates
        merged = {}
        for item in list1:
            normalized = item.lower().strip()
            if normalized not in merged:
                merged[normalized] = item

        for item in list2:
            normalized = item.lower().strip()
            if normalized not in merged:
                merged[normalized] = item

        return list(merged.values())

    @property
    def extractor_name(self) -> str:
        """Return the extractor identifier."""
        return self.mode
