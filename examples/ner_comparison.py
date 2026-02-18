"""
NER Comparison Example
Demonstrates: Comparing GLiNER, spaCy, and ensemble mode entity extraction

This example shows:
1. Entity extraction using GLiNER
2. Entity extraction using spaCy
3. Entity extraction using ensemble mode (both)
4. Side-by-side comparison of results
5. Confidence scores and unique entities analysis
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

from core.document_processing.ner_router import NERRouter
from config.settings import settings

console = Console()


# Sample text with various entity types
SAMPLE_TEXT = """
Dr. Emily Rodriguez, a renowned neuroscientist at Harvard Medical School, announced 
groundbreaking research on Alzheimer's disease at the International Conference on 
Neuroscience in Boston, Massachusetts. The study, funded by the National Institutes 
of Health with a $2.5 million grant, began in January 2022.

The research team, including Professor Michael Chen from Stanford University and 
Dr. Sarah Johnson from Johns Hopkins University, analyzed brain scans from over 
1,000 patients. Their findings, published in Nature Neuroscience on March 15, 2024, 
show promising results for early detection.

Major pharmaceutical companies like Pfizer and Moderna have expressed interest in 
developing treatments based on this research. The FDA is expected to review the 
preliminary data in Q3 2024. Dr. Rodriguez stated, "This could revolutionize how 
we approach neurodegenerative diseases."

The project, headquartered in Cambridge, Massachusetts, collaborates with research 
institutions across the United States, including MIT, UCLA, and the Mayo Clinic.
"""


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def print_entities_table(entities: List[Dict[str, Any]], title: str) -> None:
    """Print entities in a formatted table."""
    if not entities:
        console.print(f"[dim]No entities found[/dim]")
        return
    
    table = Table(title=title, show_header=True)
    table.add_column("Entity", style="green", width=30)
    table.add_column("Type", style="cyan", width=20)
    table.add_column("Confidence", style="yellow", width=12)
    table.add_column("Source", style="magenta", width=15)
    
    # Sort by confidence (descending)
    sorted_entities = sorted(entities, key=lambda x: x.get('confidence', 0), reverse=True)
    
    for entity in sorted_entities:
        sources = ", ".join(entity.get('sources', ['unknown']))
        confidence = entity.get('confidence', 0.0)
        table.add_row(
            entity['text'],
            entity['label'],
            f"{confidence:.3f}",
            sources
        )
    
    console.print(table)


def get_entity_set(entities: List[Dict[str, Any]]) -> Set[tuple]:
    """Convert entities to a set of (text, label) tuples for comparison."""
    return {(e['text'].lower(), e['label']) for e in entities}


def analyze_entity_overlap(
    gliner_entities: List[Dict[str, Any]],
    spacy_entities: List[Dict[str, Any]],
    ensemble_entities: List[Dict[str, Any]]
) -> None:
    """Analyze and display entity overlap between extractors."""
    print_section_header("ENTITY OVERLAP ANALYSIS")
    
    gliner_set = get_entity_set(gliner_entities)
    spacy_set = get_entity_set(spacy_entities)
    ensemble_set = get_entity_set(ensemble_entities)
    
    # Calculate overlaps
    both_found = gliner_set & spacy_set
    only_gliner = gliner_set - spacy_set
    only_spacy = spacy_set - gliner_set
    
    # Statistics table
    stats_table = Table(title="Extraction Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan", width=35)
    stats_table.add_column("Count", style="yellow", width=10)
    
    stats_table.add_row("Total unique entities (GLiNER)", str(len(gliner_set)))
    stats_table.add_row("Total unique entities (spaCy)", str(len(spacy_set)))
    stats_table.add_row("Total unique entities (Ensemble)", str(len(ensemble_set)))
    stats_table.add_row("", "")
    stats_table.add_row("Found by both extractors", str(len(both_found)))
    stats_table.add_row("Found only by GLiNER", str(len(only_gliner)))
    stats_table.add_row("Found only by spaCy", str(len(only_spacy)))
    
    console.print(stats_table)
    
    # Show entities found by both
    if both_found:
        console.print("\n[bold green]Entities found by BOTH extractors:[/bold green]")
        both_table = Table(show_header=True, box=None)
        both_table.add_column("Entity", style="green")
        both_table.add_column("Type", style="cyan")
        
        for text, label in sorted(both_found):
            both_table.add_row(text.title(), label)
        
        console.print(both_table)
    
    # Show unique entities
    if only_gliner:
        console.print("\n[bold magenta]Entities found ONLY by GLiNER:[/bold magenta]")
        gliner_table = Table(show_header=True, box=None)
        gliner_table.add_column("Entity", style="green")
        gliner_table.add_column("Type", style="cyan")
        
        for text, label in sorted(only_gliner):
            gliner_table.add_row(text.title(), label)
        
        console.print(gliner_table)
    
    if only_spacy:
        console.print("\n[bold blue]Entities found ONLY by spaCy:[/bold blue]")
        spacy_table = Table(show_header=True, box=None)
        spacy_table.add_column("Entity", style="green")
        spacy_table.add_column("Type", style="cyan")
        
        for text, label in sorted(only_spacy):
            spacy_table.add_row(text.title(), label)
        
        console.print(spacy_table)


def compare_confidence_scores(
    gliner_entities: List[Dict[str, Any]],
    spacy_entities: List[Dict[str, Any]]
) -> None:
    """Compare confidence scores between extractors."""
    print_section_header("CONFIDENCE SCORE COMPARISON")
    
    # Calculate average confidence
    gliner_confidences = [e.get('confidence', 0) for e in gliner_entities]
    spacy_confidences = [e.get('confidence', 0) for e in spacy_entities]
    
    avg_gliner = sum(gliner_confidences) / len(gliner_confidences) if gliner_confidences else 0
    avg_spacy = sum(spacy_confidences) / len(spacy_confidences) if spacy_confidences else 0
    
    conf_table = Table(title="Confidence Statistics", show_header=True)
    conf_table.add_column("Extractor", style="cyan")
    conf_table.add_column("Avg Confidence", style="yellow")
    conf_table.add_column("Min", style="green")
    conf_table.add_column("Max", style="green")
    
    conf_table.add_row(
        "GLiNER",
        f"{avg_gliner:.3f}",
        f"{min(gliner_confidences):.3f}" if gliner_confidences else "N/A",
        f"{max(gliner_confidences):.3f}" if gliner_confidences else "N/A"
    )
    conf_table.add_row(
        "spaCy",
        f"{avg_spacy:.3f}",
        f"{min(spacy_confidences):.3f}" if spacy_confidences else "N/A",
        f"{max(spacy_confidences):.3f}" if spacy_confidences else "N/A"
    )
    
    console.print(conf_table)
    
    # Show high-confidence entities from each
    console.print("\n[bold]High-confidence entities (>0.9):[/bold]")
    
    high_conf_gliner = [e for e in gliner_entities if e.get('confidence', 0) > 0.9]
    high_conf_spacy = [e for e in spacy_entities if e.get('confidence', 0) > 0.9]
    
    console.print(f"\n[magenta]GLiNER:[/magenta] {len(high_conf_gliner)} entities")
    if high_conf_gliner:
        for e in high_conf_gliner[:5]:  # Show top 5
            console.print(f"  • {e['text']} ({e['label']}) - {e['confidence']:.3f}")
    
    console.print(f"\n[blue]spaCy:[/blue] {len(high_conf_spacy)} entities")
    if high_conf_spacy:
        for e in high_conf_spacy[:5]:  # Show top 5
            console.print(f"  • {e['text']} ({e['label']}) - {e['confidence']:.3f}")


def analyze_entity_types(entities: List[Dict[str, Any]], extractor_name: str) -> None:
    """Analyze entity type distribution."""
    type_counts: Dict[str, int] = {}
    for entity in entities:
        label = entity['label']
        type_counts[label] = type_counts.get(label, 0) + 1
    
    if not type_counts:
        return
    
    console.print(f"\n[bold]{extractor_name} - Entity Type Distribution:[/bold]")
    
    type_table = Table(show_header=True, box=None)
    type_table.add_column("Entity Type", style="cyan")
    type_table.add_column("Count", style="yellow")
    type_table.add_column("Percentage", style="green")
    
    total = sum(type_counts.values())
    for label, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        type_table.add_row(label, str(count), f"{percentage:.1f}%")
    
    console.print(type_table)


def main():
    """Run NER comparison demonstration."""
    console.print(Panel.fit(
        "[bold magenta]NER Extractor Comparison[/bold magenta]\n"
        "Comparing GLiNER, spaCy, and Ensemble mode",
        border_style="magenta"
    ))
    
    # Display sample text
    console.print("\n[bold yellow]Sample Text:[/bold yellow]")
    console.print(Panel(SAMPLE_TEXT, border_style="blue", padding=(1, 2)))
    
    try:
        # Extract with GLiNER
        print_section_header("EXTRACTION 1: GLiNER")
        console.print("[yellow]Initializing GLiNER extractor...[/yellow]")
        
        gliner_router = NERRouter(mode="gliner")
        gliner_entities = gliner_router.extract(SAMPLE_TEXT)
        
        console.print(f"[green]✓[/green] Extracted {len(gliner_entities)} entities\n")
        print_entities_table(gliner_entities, "GLiNER Results")
        analyze_entity_types(gliner_entities, "GLiNER")
        
    except Exception as e:
        console.print(f"[red]✗[/red] GLiNER extraction failed: {str(e)}")
        gliner_entities = []
    
    try:
        # Extract with spaCy
        print_section_header("EXTRACTION 2: spaCy")
        console.print("[yellow]Initializing spaCy extractor...[/yellow]")
        
        spacy_router = NERRouter(mode="spacy")
        spacy_entities = spacy_router.extract(SAMPLE_TEXT)
        
        console.print(f"[green]✓[/green] Extracted {len(spacy_entities)} entities\n")
        print_entities_table(spacy_entities, "spaCy Results")
        analyze_entity_types(spacy_entities, "spaCy")
        
    except Exception as e:
        console.print(f"[red]✗[/red] spaCy extraction failed: {str(e)}")
        console.print("[yellow]Note: Install spaCy model: python -m spacy download en_core_web_sm[/yellow]")
        spacy_entities = []
    
    try:
        # Extract with Ensemble
        print_section_header("EXTRACTION 3: ENSEMBLE MODE")
        console.print("[yellow]Initializing ensemble extractor (GLiNER + spaCy)...[/yellow]")
        
        ensemble_router = NERRouter(mode="ensemble")
        ensemble_entities = ensemble_router.extract(SAMPLE_TEXT)
        
        console.print(f"[green]✓[/green] Extracted {len(ensemble_entities)} entities\n")
        print_entities_table(ensemble_entities, "Ensemble Results")
        analyze_entity_types(ensemble_entities, "Ensemble")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Ensemble extraction failed: {str(e)}")
        ensemble_entities = []
    
    # Comparative analysis
    if gliner_entities and spacy_entities:
        analyze_entity_overlap(gliner_entities, spacy_entities, ensemble_entities)
        compare_confidence_scores(gliner_entities, spacy_entities)
    
    # Final recommendations
    print_section_header("RECOMMENDATIONS")
    
    console.print("""
[bold green]When to use each extractor:[/bold green]

[cyan]GLiNER:[/cyan]
  • Zero-shot entity extraction with custom entity types
  • Flexible entity definitions without retraining
  • Good for domain-specific entities
  • Generally higher confidence scores

[blue]spaCy:[/blue]
  • Fast and efficient for standard entity types
  • Well-established and reliable
  • Good for PERSON, ORG, GPE, DATE entities
  • Lower memory footprint

[magenta]Ensemble:[/magenta]
  • Best recall - finds entities from both extractors
  • Combines strengths of both approaches
  • Provides source attribution for each entity
  • Recommended for comprehensive extraction
  • Slightly slower due to running both extractors

[bold yellow]Performance Tips:[/bold yellow]
  • Use GLiNER for custom entity types
  • Use spaCy for speed with standard entities
  • Use Ensemble for maximum coverage
  • Filter by confidence threshold for precision
    """)
    
    console.print("[bold green]✓ NER comparison complete![/bold green]")


if __name__ == "__main__":
    main()
