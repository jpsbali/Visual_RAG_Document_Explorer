"""
spaCy-based Named Entity Recognition extractor.

Uses spaCy transformer models for accurate NER extraction.
"""

import spacy
from spacy.tokens import Doc

from config.models import NEREntities
from config.settings import Settings


class SpacyExtractor:
    """spaCy-based NER extractor."""

    # spaCy label to our schema mapping
    LABEL_MAPPING = {
        "PERSON": "people",
        "ORG": "organizations",
        "DATE": "dates",
        "TIME": "dates",
        "GPE": "locations",  # Geopolitical entity
        "LOC": "locations",
        "FAC": "locations",  # Facility
        "MONEY": "custom",
        "PERCENT": "custom",
        "PRODUCT": "custom",
        "EVENT": "custom",
        "WORK_OF_ART": "custom",
        "LAW": "custom",
        "LANGUAGE": "custom",
        "NORP": "custom",  # Nationalities or religious/political groups
    }

    def __init__(self, settings: Settings):
        """
        Initialize spaCy extractor.

        Args:
            settings: Application settings with model config
        """
        self.settings = settings
        try:
            self.nlp = spacy.load(settings.spacy_model)
        except OSError:
            raise ValueError(
                f"spaCy model '{settings.spacy_model}' not found. "
                f"Please install it with: python -m spacy download {settings.spacy_model}"
            )

    def extract(self, text: str, extract_noun_chunks: bool = True) -> NEREntities:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from
            extract_noun_chunks: Whether to extract noun chunks as topics

        Returns:
            NEREntities object with extracted entities
        """
        # Process text with spaCy
        doc = self.nlp(text)

        # Organize entities by type
        people = []
        organizations = []
        dates = []
        locations = []
        topics = []
        custom = {}
        confidence_scores = {}

        # Extract named entities
        for ent in doc.ents:
            entity_text = ent.text
            entity_label = ent.label_
            normalized_text = self._normalize_entity(entity_text)

            # Map spaCy label to our schema
            if entity_label in self.LABEL_MAPPING:
                target_category = self.LABEL_MAPPING[entity_label]

                if target_category == "people":
                    if normalized_text not in people:
                        people.append(normalized_text)
                        confidence_scores[normalized_text] = 0.9  # spaCy doesn't provide scores

                elif target_category == "organizations":
                    if normalized_text not in organizations:
                        organizations.append(normalized_text)
                        confidence_scores[normalized_text] = 0.9

                elif target_category == "dates":
                    if normalized_text not in dates:
                        dates.append(normalized_text)
                        confidence_scores[normalized_text] = 0.9

                elif target_category == "locations":
                    if normalized_text not in locations:
                        locations.append(normalized_text)
                        confidence_scores[normalized_text] = 0.9

                elif target_category == "custom":
                    # Store in custom dict with spaCy label
                    label_key = entity_label.lower()
                    if label_key not in custom:
                        custom[label_key] = []
                    if normalized_text not in custom[label_key]:
                        custom[label_key].append(normalized_text)
                        confidence_scores[f"{label_key}:{normalized_text}"] = 0.9

        # Extract noun chunks as topics
        if extract_noun_chunks:
            for chunk in doc.noun_chunks:
                # Filter out single-word chunks and very long chunks
                chunk_text = chunk.text.strip()
                if 2 <= len(chunk_text.split()) <= 5:
                    normalized_chunk = self._normalize_entity(chunk_text)
                    if normalized_chunk not in topics:
                        topics.append(normalized_chunk)
                        confidence_scores[f"topic:{normalized_chunk}"] = 0.7

        # Limit topics to top 10 by length (longer phrases are usually more specific)
        if len(topics) > 10:
            topics = sorted(topics, key=lambda x: len(x), reverse=True)[:10]

        return NEREntities(
            people=people,
            organizations=organizations,
            dates=dates,
            locations=locations,
            topics=topics,
            custom=custom,
            extractor="spacy",
            confidence_scores=confidence_scores,
        )

    def extract_batch(
        self, texts: list[str], extract_noun_chunks: bool = True
    ) -> list[NEREntities]:
        """
        Extract entities from multiple texts efficiently.

        Args:
            texts: List of texts to extract entities from
            extract_noun_chunks: Whether to extract noun chunks as topics

        Returns:
            List of NEREntities objects
        """
        # Use spaCy's pipe for efficient batch processing
        docs = list(self.nlp.pipe(texts))

        results = []
        for doc in docs:
            # Convert Doc to text and extract
            # We need to reconstruct the extraction logic for each doc
            entities = self._extract_from_doc(doc, extract_noun_chunks)
            results.append(entities)

        return results

    def _extract_from_doc(self, doc: Doc, extract_noun_chunks: bool = True) -> NEREntities:
        """
        Extract entities from a spaCy Doc object.

        Args:
            doc: spaCy Doc object
            extract_noun_chunks: Whether to extract noun chunks as topics

        Returns:
            NEREntities object
        """
        people = []
        organizations = []
        dates = []
        locations = []
        topics = []
        custom = {}
        confidence_scores = {}

        # Extract named entities
        for ent in doc.ents:
            entity_text = ent.text
            entity_label = ent.label_
            normalized_text = self._normalize_entity(entity_text)

            if entity_label in self.LABEL_MAPPING:
                target_category = self.LABEL_MAPPING[entity_label]

                if target_category == "people" and normalized_text not in people:
                    people.append(normalized_text)
                    confidence_scores[normalized_text] = 0.9

                elif target_category == "organizations" and normalized_text not in organizations:
                    organizations.append(normalized_text)
                    confidence_scores[normalized_text] = 0.9

                elif target_category == "dates" and normalized_text not in dates:
                    dates.append(normalized_text)
                    confidence_scores[normalized_text] = 0.9

                elif target_category == "locations" and normalized_text not in locations:
                    locations.append(normalized_text)
                    confidence_scores[normalized_text] = 0.9

                elif target_category == "custom":
                    label_key = entity_label.lower()
                    if label_key not in custom:
                        custom[label_key] = []
                    if normalized_text not in custom[label_key]:
                        custom[label_key].append(normalized_text)
                        confidence_scores[f"{label_key}:{normalized_text}"] = 0.9

        # Extract noun chunks as topics
        if extract_noun_chunks:
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if 2 <= len(chunk_text.split()) <= 5:
                    normalized_chunk = self._normalize_entity(chunk_text)
                    if normalized_chunk not in topics:
                        topics.append(normalized_chunk)
                        confidence_scores[f"topic:{normalized_chunk}"] = 0.7

        if len(topics) > 10:
            topics = sorted(topics, key=lambda x: len(x), reverse=True)[:10]

        return NEREntities(
            people=people,
            organizations=organizations,
            dates=dates,
            locations=locations,
            topics=topics,
            custom=custom,
            extractor="spacy",
            confidence_scores=confidence_scores,
        )

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
        return "spacy"
