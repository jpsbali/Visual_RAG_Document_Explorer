"""
Document Summarization Example
Demonstrates: LLM-based summarization for short and long documents

This example shows:
1. Direct summarization for short documents
2. Map-reduce summarization for long documents
3. Token counting and strategy selection
4. Formatted summary output
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from core.document_processing.summarizer import DocumentSummarizer
from core.llm.llm_router import get_llm_provider
from config.settings import settings

console = Console()


# Short document example
SHORT_DOCUMENT = """
The Rise of Quantum Computing

Quantum computing represents a paradigm shift in computational technology. Unlike classical 
computers that use bits (0s and 1s), quantum computers use quantum bits or qubits, which 
can exist in multiple states simultaneously through a phenomenon called superposition.

Major tech companies including IBM, Google, and Microsoft are racing to achieve quantum 
supremacy - the point at which quantum computers can solve problems that classical computers 
cannot solve in any reasonable timeframe. In 2019, Google claimed to have achieved this 
milestone with their Sycamore processor, completing a calculation in 200 seconds that would 
take the world's fastest supercomputer 10,000 years.

The potential applications of quantum computing are vast: drug discovery, cryptography, 
financial modeling, climate simulation, and artificial intelligence. However, significant 
challenges remain, including maintaining quantum coherence and error correction.
"""


# Long document example
LONG_DOCUMENT = """
The Evolution and Impact of Artificial Intelligence in Modern Society

Introduction

Artificial Intelligence (AI) has evolved from a theoretical concept in the 1950s to one of 
the most transformative technologies of the 21st century. This document explores the history, 
current state, and future implications of AI across various sectors of society.

Historical Development

The term "artificial intelligence" was coined by John McCarthy in 1956 at the Dartmouth 
Conference, marking the birth of AI as a field of study. Early AI research focused on 
symbolic reasoning and problem-solving. The 1960s and 1970s saw the development of expert 
systems, which could mimic human decision-making in specific domains.

The AI winter of the 1980s and 1990s occurred when progress slowed due to limited 
computational power and overpromised capabilities. However, the field experienced a 
renaissance in the 2000s with the advent of machine learning, particularly deep learning, 
enabled by increased computational power and vast amounts of data.

Current Applications

Healthcare: AI is revolutionizing medical diagnostics, drug discovery, and personalized 
medicine. Machine learning algorithms can analyze medical images with accuracy matching or 
exceeding human radiologists. AI-powered systems assist in early detection of diseases like 
cancer, diabetes, and heart conditions.

Finance: The financial sector uses AI for fraud detection, algorithmic trading, risk 
assessment, and customer service. AI systems can process millions of transactions in 
real-time, identifying suspicious patterns that might indicate fraudulent activity.

Transportation: Autonomous vehicles represent one of the most visible applications of AI. 
Companies like Tesla, Waymo, and Cruise are developing self-driving cars that use computer 
vision, sensor fusion, and deep learning to navigate complex environments.

Education: AI-powered adaptive learning platforms personalize educational content based on 
individual student needs. Intelligent tutoring systems provide customized feedback and 
support, helping students learn at their own pace.

Manufacturing: AI optimizes production processes, predicts equipment failures, and enables 
quality control through computer vision. Smart factories use AI to increase efficiency and 
reduce waste.

Ethical Considerations

The rapid advancement of AI raises important ethical questions. Bias in AI systems can 
perpetuate or amplify existing societal inequalities. Privacy concerns arise from AI's 
ability to process vast amounts of personal data. The potential for job displacement due 
to automation requires careful consideration and policy responses.

Transparency and explainability in AI decision-making are crucial, especially in high-stakes 
applications like healthcare and criminal justice. The development of AI governance 
frameworks and regulations is essential to ensure responsible AI deployment.

Future Prospects

The future of AI holds both promise and challenges. Advances in natural language processing 
are making human-computer interaction more natural and intuitive. Quantum computing may 
unlock new possibilities for AI algorithms. The development of artificial general 
intelligence (AGI) - AI systems with human-like cognitive abilities - remains a long-term 
goal with profound implications.

Climate change mitigation, scientific discovery, and space exploration are areas where AI 
could make significant contributions. However, ensuring that AI benefits all of humanity 
requires international cooperation, ethical guidelines, and inclusive development practices.

Conclusion

Artificial Intelligence is reshaping our world in fundamental ways. While challenges remain, 
the potential benefits of AI are immense. Success will depend on our ability to develop and 
deploy AI responsibly, addressing ethical concerns while harnessing its transformative power 
for the betterment of society.
"""


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)."""
    return len(text) // 4


def print_document_stats(title: str, text: str) -> None:
    """Print statistics about the document."""
    word_count = len(text.split())
    char_count = len(text)
    estimated_tokens = estimate_tokens(text)
    
    stats_table = Table(title=f"{title} - Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Characters", f"{char_count:,}")
    stats_table.add_row("Words", f"{word_count:,}")
    stats_table.add_row("Estimated Tokens", f"{estimated_tokens:,}")
    
    console.print(stats_table)


def print_summary_result(summary: str, strategy: str, metadata: Dict[str, Any]) -> None:
    """Print the summary result with metadata."""
    console.print(f"\n[bold green]Strategy Used:[/bold green] {strategy}")
    
    if metadata:
        meta_table = Table(show_header=True, box=None)
        meta_table.add_column("Metadata", style="cyan")
        meta_table.add_column("Value", style="yellow")
        
        for key, value in metadata.items():
            meta_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(meta_table)
    
    console.print("\n[bold green]Summary:[/bold green]")
    console.print(Panel(summary, border_style="green", padding=(1, 2)))


def demonstrate_short_document():
    """Demonstrate direct summarization for a short document."""
    print_section_header("EXAMPLE 1: SHORT DOCUMENT SUMMARIZATION")
    
    console.print("[yellow]Document: The Rise of Quantum Computing[/yellow]\n")
    print_document_stats("Short Document", SHORT_DOCUMENT)
    
    console.print("\n[yellow]Initializing summarizer...[/yellow]")
    
    try:
        summarizer = DocumentSummarizer()
        
        console.print("[yellow]Generating summary (direct approach)...[/yellow]")
        result = summarizer.summarize(SHORT_DOCUMENT, max_length=150)
        
        print_summary_result(
            result['summary'],
            result.get('strategy', 'direct'),
            result.get('metadata', {})
        )
        
    except Exception as e:
        console.print(f"[red]✗[/red] Summarization failed: {str(e)}")
        console.print("[yellow]Note: Ensure LLM API keys are configured in .env file[/yellow]")
        console.print("[yellow]Required: OPENAI_API_KEY or OPENROUTER_API_KEY[/yellow]")


def demonstrate_long_document():
    """Demonstrate map-reduce summarization for a long document."""
    print_section_header("EXAMPLE 2: LONG DOCUMENT SUMMARIZATION")
    
    console.print("[yellow]Document: The Evolution and Impact of AI in Modern Society[/yellow]\n")
    print_document_stats("Long Document", LONG_DOCUMENT)
    
    console.print("\n[yellow]Initializing summarizer...[/yellow]")
    
    try:
        summarizer = DocumentSummarizer()
        
        console.print("[yellow]Generating summary (map-reduce approach)...[/yellow]")
        console.print("[dim]This may take longer as the document is processed in chunks...[/dim]\n")
        
        result = summarizer.summarize(LONG_DOCUMENT, max_length=300)
        
        print_summary_result(
            result['summary'],
            result.get('strategy', 'map-reduce'),
            result.get('metadata', {})
        )
        
    except Exception as e:
        console.print(f"[red]✗[/red] Summarization failed: {str(e)}")
        console.print("[yellow]Note: Ensure LLM API keys are configured in .env file[/yellow]")


def demonstrate_custom_parameters():
    """Demonstrate summarization with custom parameters."""
    print_section_header("EXAMPLE 3: CUSTOM SUMMARIZATION PARAMETERS")
    
    console.print("[yellow]Using custom parameters for summarization[/yellow]\n")
    
    try:
        summarizer = DocumentSummarizer()
        
        # Very brief summary
        console.print("[bold]Brief Summary (50 words):[/bold]")
        result_brief = summarizer.summarize(SHORT_DOCUMENT, max_length=50)
        console.print(Panel(result_brief['summary'], border_style="blue"))
        
        # Detailed summary
        console.print("\n[bold]Detailed Summary (200 words):[/bold]")
        result_detailed = summarizer.summarize(SHORT_DOCUMENT, max_length=200)
        console.print(Panel(result_detailed['summary'], border_style="blue"))
        
        # Compare lengths
        console.print("\n[bold green]Comparison:[/bold green]")
        compare_table = Table(show_header=True)
        compare_table.add_column("Type", style="cyan")
        compare_table.add_column("Word Count", style="yellow")
        compare_table.add_column("Compression Ratio", style="green")
        
        original_words = len(SHORT_DOCUMENT.split())
        brief_words = len(result_brief['summary'].split())
        detailed_words = len(result_detailed['summary'].split())
        
        compare_table.add_row(
            "Original",
            str(original_words),
            "100%"
        )
        compare_table.add_row(
            "Brief Summary",
            str(brief_words),
            f"{(brief_words/original_words)*100:.1f}%"
        )
        compare_table.add_row(
            "Detailed Summary",
            str(detailed_words),
            f"{(detailed_words/original_words)*100:.1f}%"
        )
        
        console.print(compare_table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Custom summarization failed: {str(e)}")


def main():
    """Run all summarization demonstrations."""
    console.print(Panel.fit(
        "[bold magenta]Document Summarization Examples[/bold magenta]\n"
        "Demonstrating direct and map-reduce summarization strategies",
        border_style="magenta"
    ))
    
    # Check if LLM provider is available
    try:
        llm_provider = get_llm_provider()
        console.print(f"[green]✓[/green] LLM provider initialized: {llm_provider.__class__.__name__}\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize LLM provider: {str(e)}")
        console.print("[yellow]Please configure API keys in .env file[/yellow]")
        console.print("[yellow]Required: OPENAI_API_KEY or OPENROUTER_API_KEY[/yellow]")
        sys.exit(1)
    
    # Run demonstrations
    demonstrate_short_document()
    demonstrate_long_document()
    demonstrate_custom_parameters()
    
    # Final summary
    print_section_header("SUMMARY")
    
    console.print("""
[bold green]Key Takeaways:[/bold green]

1. [cyan]Direct Summarization:[/cyan] Used for short documents that fit within the LLM's context window
2. [cyan]Map-Reduce Summarization:[/cyan] Used for long documents, processes in chunks then combines
3. [cyan]Automatic Strategy Selection:[/cyan] The summarizer automatically chooses the best approach
4. [cyan]Customizable Length:[/cyan] Control summary length with the max_length parameter
5. [cyan]Metadata Tracking:[/cyan] Get information about tokens used and processing strategy

[bold yellow]Tips:[/bold yellow]
- Adjust max_length based on your needs (shorter = more concise, longer = more detailed)
- Map-reduce is slower but handles documents of any length
- Check metadata for token usage and cost estimation
    """)
    
    console.print("[bold green]✓ All summarization examples complete![/bold green]")


if __name__ == "__main__":
    main()
