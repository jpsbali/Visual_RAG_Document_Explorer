"""
Basic Pipeline Example
Demonstrates: Document loading, chunking, NER extraction, and embedding generation

This example shows the complete Phase 1B pipeline:
1. Load a sample document
2. Chunk the document with AdaptiveChunker
3. Extract entities using NER router (ensemble mode)
4. Generate embeddings using embedding router
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from core.document_processing.loaders import DocumentLoader
from core.document_processing.chunker import AdaptiveChunker
from core.document_processing.ner_router import NERRouter
from core.embeddings.embedding_router import get_embedding_service
from config.settings import settings

console = Console()


# Sample document text
SAMPLE_TEXT = """
Artificial Intelligence and Machine Learning in Healthcare

The integration of artificial intelligence (AI) and machine learning (ML) technologies 
in healthcare has revolutionized medical diagnostics and patient care. Dr. Sarah Johnson, 
a leading researcher at Stanford University, has been pioneering work in this field since 2018.

Recent studies published in Nature Medicine demonstrate that AI algorithms can detect 
early-stage cancer with 95% accuracy. The research team, led by Professor Michael Chen 
at MIT, developed a deep learning model trained on over 100,000 medical images.

Major technology companies like Google Health and IBM Watson are investing heavily in 
healthcare AI. In 2023, Google announced a $500 million initiative to develop AI-powered 
diagnostic tools. The project, based in Mountain View, California, aims to make advanced 
medical diagnostics accessible to underserved communities.

The FDA approved the first AI-based diagnostic system in January 2024, marking a 
significant milestone in medical technology. This system, developed by MedTech Solutions, 
can analyze chest X-rays and detect pneumonia in under 10 seconds.

However, challenges remain. Dr. Emily Rodriguez from Johns Hopkins University warns 
about potential biases in AI training data. "We must ensure that AI systems are trained 
on diverse patient populations to avoid perpetuating healthcare disparities," she stated 
in a recent interview with The New England Journal of Medicine.
"""


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def print_document_info(text: str) -> None:
    """Print information about the loaded document."""
    print_section_header("STEP 1: DOCUMENT LOADING")
    
    word_count = len(text.split())
    char_count = len(text)
    line_count = len(text.split('\n'))
    
    info_table = Table(title="Document Statistics", show_header=True)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Characters", str(char_count))
    info_table.add_row("Words", str(word_count))
    info_table.add_row("Lines", str(line_count))
    
    console.print(info_table)
    console.print("\n[yellow]Sample preview (first 200 chars):[/yellow]")
    console.print(Panel(text[:200] + "...", border_style="blue"))


def print_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Print information about generated chunks."""
    print_section_header("STEP 2: DOCUMENT CHUNKING")
    
    console.print(f"[green]✓[/green] Generated [bold]{len(chunks)}[/bold] chunks\n")
    
    chunks_table = Table(title="Chunk Details", show_header=True)
    chunks_table.add_column("Chunk #", style="cyan", width=10)
    chunks_table.add_column("Tokens", style="yellow", width=10)
    chunks_table.add_column("Preview", style="white", width=60)
    
    for i, chunk in enumerate(chunks, 1):
        preview = chunk['text'][:60].replace('\n', ' ') + "..."
        chunks_table.add_row(
            str(i),
            str(chunk.get('token_count', 'N/A')),
            preview
        )
    
    console.print(chunks_table)


def print_entities(entities: List[Dict[str, Any]], chunk_idx: int) -> None:
    """Print extracted entities for a chunk."""
    if not entities:
        console.print(f"  [dim]No entities found in chunk {chunk_idx}[/dim]")
        return
    
    entities_table = Table(show_header=True, box=None, padding=(0, 2))
    entities_table.add_column("Entity", style="green")
    entities_table.add_column("Type", style="cyan")
    entities_table.add_column("Confidence", style="yellow")
    entities_table.add_column("Sources", style="magenta")
    
    for entity in entities:
        sources = ", ".join(entity.get('sources', ['unknown']))
        confidence = entity.get('confidence', 0.0)
        entities_table.add_row(
            entity['text'],
            entity['label'],
            f"{confidence:.2f}",
            sources
        )
    
    console.print(entities_table)


def print_embeddings(embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> None:
    """Print information about generated embeddings."""
    print_section_header("STEP 4: EMBEDDING GENERATION")
    
    console.print(f"[green]✓[/green] Generated embeddings for [bold]{len(embeddings)}[/bold] chunks\n")
    
    embed_table = Table(title="Embedding Details", show_header=True)
    embed_table.add_column("Chunk #", style="cyan", width=10)
    embed_table.add_column("Dimensions", style="yellow", width=15)
    embed_table.add_column("Sample Values", style="white", width=55)
    
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks), 1):
        sample_values = ", ".join([f"{v:.4f}" for v in embedding[:5]]) + "..."
        embed_table.add_row(
            str(i),
            str(len(embedding)),
            sample_values
        )
    
    console.print(embed_table)


def main():
    """Run the complete pipeline demonstration."""
    console.print(Panel.fit(
        "[bold magenta]Phase 1B Pipeline Demonstration[/bold magenta]\n"
        "Complete workflow: Load → Chunk → Extract Entities → Embed",
        border_style="magenta"
    ))
    
    try:
        # Step 1: Load document
        print_document_info(SAMPLE_TEXT)
        
        # Step 2: Chunk the document
        console.print("\n[yellow]Initializing AdaptiveChunker...[/yellow]")
        chunker = AdaptiveChunker(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50
        )
        
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        print_chunks(chunks)
        
        # Step 3: Extract entities using NER router (ensemble mode)
        print_section_header("STEP 3: ENTITY EXTRACTION (ENSEMBLE MODE)")
        
        console.print("[yellow]Initializing NER Router with ensemble mode...[/yellow]")
        console.print("[dim]This may take a moment to load models...[/dim]\n")
        
        try:
            ner_router = NERRouter(mode="ensemble")
            
            for i, chunk in enumerate(chunks, 1):
                console.print(f"\n[bold]Chunk {i}:[/bold]")
                entities = ner_router.extract(chunk['text'])
                print_entities(entities, i)
                
                # Add entities to chunk metadata
                chunk['entities'] = entities
                
        except Exception as e:
            console.print(f"[red]✗[/red] NER extraction failed: {str(e)}")
            console.print("[yellow]Note: Ensure spaCy model is installed: python -m spacy download en_core_web_sm[/yellow]")
            console.print("[yellow]Continuing without entity extraction...[/yellow]")
        
        # Step 4: Generate embeddings
        console.print("\n[yellow]Initializing embedding service...[/yellow]")
        
        try:
            embedding_service = get_embedding_service()
            
            # Extract text from chunks for embedding
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            console.print(f"[yellow]Generating embeddings for {len(chunk_texts)} chunks...[/yellow]")
            embeddings = embedding_service.embed_batch(chunk_texts)
            
            print_embeddings(embeddings, chunks)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
                
        except Exception as e:
            console.print(f"[red]✗[/red] Embedding generation failed: {str(e)}")
            console.print("[yellow]Note: Check your API keys in .env file[/yellow]")
            console.print("[yellow]Required: VOYAGE_API_KEY or ensure BGE-M3 model is available[/yellow]")
        
        # Final summary
        print_section_header("PIPELINE SUMMARY")
        
        summary_table = Table(show_header=True)
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Output", style="yellow")
        
        summary_table.add_row("Document Loading", "✓ Complete", "1 document loaded")
        summary_table.add_row("Chunking", "✓ Complete", f"{len(chunks)} chunks created")
        
        total_entities = sum(len(chunk.get('entities', [])) for chunk in chunks)
        entity_status = "✓ Complete" if total_entities > 0 else "⚠ Skipped"
        summary_table.add_row("Entity Extraction", entity_status, f"{total_entities} entities found")
        
        has_embeddings = any('embedding' in chunk for chunk in chunks)
        embed_status = "✓ Complete" if has_embeddings else "⚠ Skipped"
        embed_count = sum(1 for chunk in chunks if 'embedding' in chunk)
        summary_table.add_row("Embedding Generation", embed_status, f"{embed_count} embeddings created")
        
        console.print(summary_table)
        
        console.print("\n[bold green]✓ Pipeline demonstration complete![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
