"""
Batch Processing Example
Demonstrates: Processing multiple documents efficiently in batches

This example shows:
1. Loading multiple sample documents
2. Batch chunking all documents
3. Batch entity extraction
4. Batch embedding generation
5. Processing statistics and performance metrics
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from core.document_processing.chunker import AdaptiveChunker
from core.document_processing.ner_router import NERRouter
from core.embeddings.embedding_router import get_embedding_service
from config.settings import settings

console = Console()


# Sample documents for batch processing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Climate Change Research",
        "text": """
        Recent studies by Dr. James Wilson at the University of California, Berkeley, 
        show alarming trends in global temperature rise. The research, published in 
        Science Magazine in December 2023, analyzed climate data from 1950 to 2023. 
        The team found that average global temperatures have increased by 1.2°C since 
        pre-industrial times. Major cities like New York, London, and Tokyo are 
        experiencing more frequent extreme weather events. The United Nations Climate 
        Summit in Paris emphasized the need for immediate action.
        """
    },
    {
        "id": "doc_002",
        "title": "Advances in Renewable Energy",
        "text": """
        Tesla and SolarCity have announced a breakthrough in solar panel efficiency. 
        The new technology, developed by engineers at MIT and Stanford University, 
        achieves 35% efficiency, a significant improvement over the industry standard 
        of 20%. Elon Musk stated that this could reduce solar installation costs by 
        40%. The technology will be manufactured in Gigafactory 2 in Buffalo, New York. 
        The Department of Energy has allocated $100 million in funding for further 
        research. Commercial deployment is expected in Q2 2025.
        """
    },
    {
        "id": "doc_003",
        "title": "Medical Breakthrough in Cancer Treatment",
        "text": """
        Dr. Maria Garcia at Johns Hopkins Hospital has developed a novel immunotherapy 
        treatment for pancreatic cancer. Clinical trials conducted from 2021 to 2024 
        showed a 60% improvement in patient survival rates. The treatment, called 
        ImmunoBoost-7, was tested on 500 patients across 20 hospitals in the United States. 
        Pharmaceutical companies Pfizer and Merck have expressed interest in licensing 
        the technology. The FDA is expected to approve the treatment by mid-2025. 
        Dr. Garcia will present findings at the American Cancer Society conference in 
        Chicago this September.
        """
    },
    {
        "id": "doc_004",
        "title": "Artificial Intelligence in Education",
        "text": """
        Professor David Lee from Carnegie Mellon University has developed an AI-powered 
        tutoring system that adapts to individual student learning styles. The system, 
        called EduAI, uses machine learning algorithms to personalize educational content. 
        Pilot programs in schools across Pennsylvania, Ohio, and Michigan showed a 25% 
        improvement in student test scores. The Bill & Melinda Gates Foundation has 
        invested $50 million in the project. Microsoft and Google are partnering to 
        integrate the technology into their educational platforms. The system will be 
        available to 1,000 schools starting in Fall 2024.
        """
    },
    {
        "id": "doc_005",
        "title": "Space Exploration Update",
        "text": """
        NASA's Artemis III mission, scheduled for 2026, will return humans to the Moon 
        for the first time since Apollo 17 in 1972. Commander Sarah Mitchell and pilot 
        Robert Chen will spend 7 days on the lunar surface near the South Pole. The 
        mission, launched from Kennedy Space Center in Florida, aims to establish a 
        permanent lunar base. SpaceX's Starship will serve as the lunar lander. The 
        European Space Agency and Japan Aerospace Exploration Agency are contributing 
        equipment and expertise. Total mission cost is estimated at $93 billion.
        """
    }
]


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def print_document_overview() -> None:
    """Print overview of documents to be processed."""
    print_section_header("DOCUMENT OVERVIEW")
    
    doc_table = Table(title="Documents to Process", show_header=True)
    doc_table.add_column("ID", style="cyan", width=12)
    doc_table.add_column("Title", style="green", width=35)
    doc_table.add_column("Words", style="yellow", width=10)
    doc_table.add_column("Preview", style="white", width=20)
    
    for doc in SAMPLE_DOCUMENTS:
        word_count = len(doc['text'].split())
        preview = doc['text'].strip()[:20].replace('\n', ' ') + "..."
        doc_table.add_row(
            doc['id'],
            doc['title'],
            str(word_count),
            preview
        )
    
    console.print(doc_table)
    console.print(f"\n[bold]Total documents:[/bold] {len(SAMPLE_DOCUMENTS)}")


def batch_chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk all documents in batch."""
    print_section_header("STEP 1: BATCH CHUNKING")
    
    console.print("[yellow]Initializing AdaptiveChunker...[/yellow]")
    chunker = AdaptiveChunker(
        chunk_size=150,
        chunk_overlap=30,
        min_chunk_size=50
    )
    
    all_chunks = []
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Chunking documents...", total=len(documents))
        
        for doc in documents:
            chunks = chunker.chunk_text(doc['text'])
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk['doc_id'] = doc['id']
                chunk['doc_title'] = doc['title']
                all_chunks.append(chunk)
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[green]✓[/green] Chunking complete!")
    
    # Statistics
    stats_table = Table(show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    
    stats_table.add_row("Total chunks created", str(len(all_chunks)))
    stats_table.add_row("Average chunks per document", f"{len(all_chunks)/len(documents):.1f}")
    stats_table.add_row("Processing time", f"{elapsed_time:.2f}s")
    stats_table.add_row("Chunks per second", f"{len(all_chunks)/elapsed_time:.1f}")
    
    console.print(stats_table)
    
    return all_chunks


def batch_extract_entities(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract entities from all chunks in batch."""
    print_section_header("STEP 2: BATCH ENTITY EXTRACTION")
    
    console.print("[yellow]Initializing NER Router (ensemble mode)...[/yellow]")
    console.print("[dim]This may take a moment to load models...[/dim]\n")
    
    try:
        ner_router = NERRouter(mode="ensemble")
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Extracting entities...", total=len(chunks))
            
            for chunk in chunks:
                entities = ner_router.extract(chunk['text'])
                chunk['entities'] = entities
                progress.update(task, advance=1)
        
        elapsed_time = time.time() - start_time
        
        console.print(f"\n[green]✓[/green] Entity extraction complete!")
        
        # Statistics
        total_entities = sum(len(chunk.get('entities', [])) for chunk in chunks)
        unique_entities = set()
        entity_types = {}
        
        for chunk in chunks:
            for entity in chunk.get('entities', []):
                unique_entities.add((entity['text'].lower(), entity['label']))
                entity_types[entity['label']] = entity_types.get(entity['label'], 0) + 1
        
        stats_table = Table(show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Total entities extracted", str(total_entities))
        stats_table.add_row("Unique entities", str(len(unique_entities)))
        stats_table.add_row("Average entities per chunk", f"{total_entities/len(chunks):.1f}")
        stats_table.add_row("Processing time", f"{elapsed_time:.2f}s")
        stats_table.add_row("Chunks per second", f"{len(chunks)/elapsed_time:.1f}")
        
        console.print(stats_table)
        
        # Entity type distribution
        if entity_types:
            console.print("\n[bold]Entity Type Distribution:[/bold]")
            type_table = Table(show_header=True, box=None)
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="yellow")
            
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                type_table.add_row(entity_type, str(count))
            
            console.print(type_table)
        
        return chunks
        
    except Exception as e:
        console.print(f"[red]✗[/red] Entity extraction failed: {str(e)}")
        console.print("[yellow]Note: Ensure spaCy model is installed: python -m spacy download en_core_web_sm[/yellow]")
        console.print("[yellow]Continuing without entity extraction...[/yellow]")
        return chunks


def batch_generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for all chunks in batch."""
    print_section_header("STEP 3: BATCH EMBEDDING GENERATION")
    
    console.print("[yellow]Initializing embedding service...[/yellow]")
    
    try:
        embedding_service = get_embedding_service()
        
        # Extract texts for embedding
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        console.print(f"[yellow]Generating embeddings for {len(chunk_texts)} chunks...[/yellow]")
        console.print("[dim]Processing in batches for efficiency...[/dim]\n")
        
        start_time = time.time()
        
        # Generate embeddings in batch
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        elapsed_time = time.time() - start_time
        
        console.print(f"[green]✓[/green] Embedding generation complete!")
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        # Statistics
        stats_table = Table(show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Total embeddings generated", str(len(embeddings)))
        stats_table.add_row("Embedding dimensions", str(len(embeddings[0])))
        stats_table.add_row("Processing time", f"{elapsed_time:.2f}s")
        stats_table.add_row("Embeddings per second", f"{len(embeddings)/elapsed_time:.1f}")
        
        console.print(stats_table)
        
        return chunks
        
    except Exception as e:
        console.print(f"[red]✗[/red] Embedding generation failed: {str(e)}")
        console.print("[yellow]Note: Check your API keys in .env file[/yellow]")
        console.print("[yellow]Required: VOYAGE_API_KEY or ensure BGE-M3 model is available[/yellow]")
        return chunks


def print_processing_summary(chunks: List[Dict[str, Any]]) -> None:
    """Print final processing summary."""
    print_section_header("PROCESSING SUMMARY")
    
    # Group chunks by document
    doc_stats = {}
    for chunk in chunks:
        doc_id = chunk['doc_id']
        if doc_id not in doc_stats:
            doc_stats[doc_id] = {
                'title': chunk['doc_title'],
                'chunks': 0,
                'entities': 0,
                'has_embeddings': False
            }
        
        doc_stats[doc_id]['chunks'] += 1
        doc_stats[doc_id]['entities'] += len(chunk.get('entities', []))
        if 'embedding' in chunk:
            doc_stats[doc_id]['has_embeddings'] = True
    
    # Per-document summary
    summary_table = Table(title="Per-Document Summary", show_header=True)
    summary_table.add_column("Document", style="cyan", width=35)
    summary_table.add_column("Chunks", style="yellow", width=10)
    summary_table.add_column("Entities", style="green", width=10)
    summary_table.add_column("Embeddings", style="magenta", width=12)
    
    for doc_id, stats in doc_stats.items():
        summary_table.add_row(
            stats['title'],
            str(stats['chunks']),
            str(stats['entities']),
            "✓" if stats['has_embeddings'] else "✗"
        )
    
    console.print(summary_table)
    
    # Overall statistics
    total_entities = sum(stats['entities'] for stats in doc_stats.values())
    total_chunks = len(chunks)
    docs_with_embeddings = sum(1 for stats in doc_stats.values() if stats['has_embeddings'])
    
    console.print(f"\n[bold green]Overall Statistics:[/bold green]")
    console.print(f"  • Total documents processed: {len(doc_stats)}")
    console.print(f"  • Total chunks created: {total_chunks}")
    console.print(f"  • Total entities extracted: {total_entities}")
    console.print(f"  • Documents with embeddings: {docs_with_embeddings}/{len(doc_stats)}")


def main():
    """Run batch processing demonstration."""
    console.print(Panel.fit(
        "[bold magenta]Batch Processing Demonstration[/bold magenta]\n"
        "Efficiently processing multiple documents through the complete pipeline",
        border_style="magenta"
    ))
    
    # Show document overview
    print_document_overview()
    
    # Step 1: Batch chunking
    chunks = batch_chunk_documents(SAMPLE_DOCUMENTS)
    
    # Step 2: Batch entity extraction
    chunks = batch_extract_entities(chunks)
    
    # Step 3: Batch embedding generation
    chunks = batch_generate_embeddings(chunks)
    
    # Final summary
    print_processing_summary(chunks)
    
    # Performance tips
    print_section_header("PERFORMANCE TIPS")
    
    console.print("""
[bold green]Batch Processing Best Practices:[/bold green]

1. [cyan]Chunking:[/cyan]
   • Process all documents at once for consistent chunk sizes
   • Adjust chunk_size based on your embedding model's limits
   • Use chunk_overlap to maintain context between chunks

2. [cyan]Entity Extraction:[/cyan]
   • Ensemble mode provides best results but is slower
   • Use GLiNER or spaCy alone for faster processing
   • Consider filtering entities by confidence threshold

3. [cyan]Embedding Generation:[/cyan]
   • Batch embedding is much faster than individual calls
   • Most embedding services have batch size limits (check docs)
   • Consider caching embeddings for frequently accessed documents

4. [cyan]Memory Management:[/cyan]
   • For very large document sets, process in batches
   • Clear embeddings from memory after storing in vector DB
   • Monitor memory usage with large models (GLiNER, BGE-M3)

5. [cyan]Optimization:[/cyan]
   • Use multiprocessing for CPU-bound tasks (chunking, NER)
   • Use async/await for I/O-bound tasks (API calls)
   • Cache model loading (don't reload for each document)
    """)
    
    console.print("[bold green]✓ Batch processing demonstration complete![/bold green]")


if __name__ == "__main__":
    main()
