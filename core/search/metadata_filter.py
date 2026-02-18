"""
Metadata filtering for search results.

Filters chunks based on extracted NER entities and other metadata.
"""

from config.models import RetrievedChunk, ChunkMetadata
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filter search results based on metadata criteria.
    
    Supports filtering by:
    - Entity types (people, organizations, dates, locations, topics)
    - File type
    - Date range
    - Custom metadata fields
    """
    
    @staticmethod
    def apply_filters(
        chunks: list[RetrievedChunk],
        filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """
        Apply metadata filters to search results.
        
        Args:
            chunks: List of retrieved chunks
            filters: Filter criteria dict, e.g.:
                {
                    "organizations": ["Acme Corp", "TechCo"],
                    "people": ["John Doe"],
                    "file_type": ["pdf"],
                    "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
                }
                
        Returns:
            Filtered list of chunks
        """
        if not filters:
            return chunks
            
        filtered = []
        for chunk in chunks:
            if MetadataFilter._matches_filters(chunk.metadata, filters):
                filtered.append(chunk)
                
        logger.info(f"Filtered {len(chunks)} → {len(filtered)} chunks")
        return filtered
        
    @staticmethod
    def _matches_filters(metadata: ChunkMetadata, filters: dict) -> bool:
        """
        Check if metadata matches all filter criteria.
        
        Args:
            metadata: Chunk metadata
            filters: Filter criteria
            
        Returns:
            True if metadata matches all filters
        """
        for field, criteria in filters.items():
            if field == "organizations":
                if not any(org in metadata.entities.organizations for org in criteria):
                    return False
                    
            elif field == "people":
                if not any(person in metadata.entities.people for person in criteria):
                    return False
                    
            elif field == "dates":
                if not any(date in metadata.entities.dates for date in criteria):
                    return False
                    
            elif field == "locations":
                if not any(loc in metadata.entities.locations for loc in criteria):
                    return False
                    
            elif field == "topics":
                if not any(topic in metadata.entities.topics for topic in criteria):
                    return False
                    
            elif field == "file_type":
                if metadata.file_type not in criteria:
                    return False
                    
            elif field == "source_file":
                if metadata.source_file not in criteria:
                    return False
                    
            elif field == "chunk_type":
                if metadata.chunk_type not in criteria:
                    return False
                    
            elif field == "date_range":
                # Filter by document creation date
                start = criteria.get("start")
                end = criteria.get("end")
                doc_date = metadata.created_at
                
                if start:
                    start_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
                    if doc_date < start_dt:
                        return False
                if end:
                    end_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
                    if doc_date > end_dt:
                        return False
                        
        return True
        
    @staticmethod
    def build_filter_summary(filters: dict) -> str:
        """
        Build human-readable filter summary.
        
        Args:
            filters: Filter criteria
            
        Returns:
            Human-readable filter description
        """
        if not filters:
            return "No filters applied"
            
        parts = []
        for field, criteria in filters.items():
            if isinstance(criteria, list):
                parts.append(f"{field}: {', '.join(str(c) for c in criteria)}")
            elif isinstance(criteria, dict):
                if field == "date_range":
                    start = criteria.get("start", "any")
                    end = criteria.get("end", "any")
                    parts.append(f"date_range: {start} to {end}")
                else:
                    parts.append(f"{field}: {criteria}")
                    
        return " | ".join(parts)
        
    @staticmethod
    def filter_by_entity_type(
        chunks: list[RetrievedChunk],
        entity_type: str,
        entity_values: list[str]
    ) -> list[RetrievedChunk]:
        """
        Filter chunks by a specific entity type.
        
        Args:
            chunks: List of retrieved chunks
            entity_type: Entity type (organizations, people, dates, locations, topics)
            entity_values: List of entity values to match
            
        Returns:
            Filtered chunks
        """
        return MetadataFilter.apply_filters(
            chunks,
            filters={entity_type: entity_values}
        )
        
    @staticmethod
    def filter_by_file_type(
        chunks: list[RetrievedChunk],
        file_types: list[str]
    ) -> list[RetrievedChunk]:
        """
        Filter chunks by file type.
        
        Args:
            chunks: List of retrieved chunks
            file_types: List of file types (pdf, docx, txt, html, json)
            
        Returns:
            Filtered chunks
        """
        return MetadataFilter.apply_filters(
            chunks,
            filters={"file_type": file_types}
        )
        
    @staticmethod
    def filter_by_date_range(
        chunks: list[RetrievedChunk],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> list[RetrievedChunk]:
        """
        Filter chunks by date range.
        
        Args:
            chunks: List of retrieved chunks
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            Filtered chunks
        """
        date_range = {}
        if start_date:
            date_range["start"] = start_date
        if end_date:
            date_range["end"] = end_date
            
        if not date_range:
            return chunks
            
        return MetadataFilter.apply_filters(
            chunks,
            filters={"date_range": date_range}
        )
