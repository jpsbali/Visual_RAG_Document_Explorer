"""
Benchmark page for Visual RAG Document Explorer.

Provides performance benchmarking dashboard for vector databases.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from config.settings import Settings
from config.models import BenchmarkResult
from core.vectordb.benchmark import VectorDBBenchmark
from ui.components import render_sidebar


def initialize_session_state() -> None:
    """Initialize session state variables for benchmark page."""
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = []
    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False
    if "benchmark_history" not in st.session_state:
        # Try to load history from file
        history_file = Path("data/benchmark_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    st.session_state.benchmark_history = json.load(f)
            except:
                st.session_state.benchmark_history = []
        else:
            st.session_state.benchmark_history = []


def save_benchmark_history() -> None:
    """Save benchmark history to file."""
    try:
        history_file = Path("data/benchmark_history.json")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(st.session_state.benchmark_history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving benchmark history: {str(e)}")


def create_latency_chart(results: Dict[str, BenchmarkResult]) -> go.Figure:
    """
    Create latency comparison chart.
    
    Args:
        results: Dictionary mapping db_name to BenchmarkResult
        
    Returns:
        Plotly figure
    """
    # Prepare data
    db_names = []
    p50_values = []
    p95_values = []
    p99_values = []
    
    for db_name, result in results.items():
        db_names.append(db_name.capitalize())
        p50_values.append(result.latency_p50_ms)
        p95_values.append(result.latency_p95_ms)
        p99_values.append(result.latency_p99_ms)
    
    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name='p50', x=db_names, y=p50_values, marker_color='lightblue'),
        go.Bar(name='p95', x=db_names, y=p95_values, marker_color='orange'),
        go.Bar(name='p99', x=db_names, y=p99_values, marker_color='red')
    ])
    
    fig.update_layout(
        title='Query Latency Comparison (ms)',
        xaxis_title='Database',
        yaxis_title='Latency (ms)',
        barmode='group',
        height=400
    )
    
    return fig


def create_throughput_chart(results: Dict[str, BenchmarkResult]) -> go.Figure:
    """
    Create throughput comparison chart.
    
    Args:
        results: Dictionary mapping db_name to BenchmarkResult
        
    Returns:
        Plotly figure
    """
    db_names = []
    throughput_values = []
    
    for db_name, result in results.items():
        db_names.append(db_name.capitalize())
        throughput_values.append(result.throughput_ops_per_sec)
    
    fig = go.Figure(data=[
        go.Bar(x=db_names, y=throughput_values, marker_color='green')
    ])
    
    fig.update_layout(
        title='Throughput Comparison (ops/sec)',
        xaxis_title='Database',
        yaxis_title='Operations per Second',
        height=400
    )
    
    return fig


def create_recall_chart(results: Dict[str, BenchmarkResult]) -> go.Figure:
    """
    Create recall@k comparison chart.
    
    Args:
        results: Dictionary mapping db_name to BenchmarkResult
        
    Returns:
        Plotly figure
    """
    # Prepare data
    data = []
    
    for db_name, result in results.items():
        if result.recall_at_k:
            for k, recall in result.recall_at_k.items():
                data.append({
                    'Database': db_name.capitalize(),
                    'k': f'@{k}',
                    'Recall': recall
                })
    
    if not data:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title='Recall@k Comparison',
            annotations=[{
                'text': 'No recall data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='k',
        y='Recall',
        color='Database',
        barmode='group',
        title='Recall@k Comparison'
    )
    
    fig.update_layout(
        xaxis_title='Top-k',
        yaxis_title='Recall',
        height=400
    )
    
    return fig


def create_memory_chart(results: Dict[str, BenchmarkResult]) -> go.Figure:
    """
    Create memory usage comparison chart.
    
    Args:
        results: Dictionary mapping db_name to BenchmarkResult
        
    Returns:
        Plotly figure
    """
    db_names = []
    memory_values = []
    
    for db_name, result in results.items():
        db_names.append(db_name.capitalize())
        memory_values.append(result.memory_usage_mb)
    
    fig = go.Figure(data=[
        go.Bar(x=db_names, y=memory_values, marker_color='purple')
    ])
    
    fig.update_layout(
        title='Memory Usage Comparison (MB)',
        xaxis_title='Database',
        yaxis_title='Memory (MB)',
        height=400
    )
    
    return fig


def create_comparison_table(results: Dict[str, BenchmarkResult]) -> pd.DataFrame:
    """
    Create comparison table with all metrics.
    
    Args:
        results: Dictionary mapping db_name to BenchmarkResult
        
    Returns:
        Pandas DataFrame
    """
    data = []
    
    for db_name, result in results.items():
        data.append({
            'Database': db_name.capitalize(),
            'Operation': result.operation.capitalize(),
            'Documents': result.num_documents,
            'p50 Latency (ms)': f"{result.latency_p50_ms:.2f}",
            'p95 Latency (ms)': f"{result.latency_p95_ms:.2f}",
            'p99 Latency (ms)': f"{result.latency_p99_ms:.2f}",
            'Throughput (ops/s)': f"{result.throughput_ops_per_sec:.2f}",
            'Memory (MB)': f"{result.memory_usage_mb:.2f}",
            'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(data)


def render_benchmark_controls() -> Dict[str, Any]:
    """
    Render benchmark configuration controls.
    
    Returns:
        Dictionary of benchmark parameters
    """
    st.subheader("⚙️ Benchmark Configuration")
    
    params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        params["databases"] = st.multiselect(
            "Databases to Benchmark",
            options=["qdrant", "milvus"],
            default=["qdrant", "milvus"],
            help="Select which databases to benchmark"
        )
    
    with col2:
        params["num_documents"] = st.number_input(
            "Number of Documents",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of test documents to index"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        params["num_queries"] = st.number_input(
            "Number of Queries",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of search queries to run"
        )
    
    with col2:
        params["dimension"] = st.selectbox(
            "Vector Dimension",
            options=[384, 768, 1024, 1536],
            index=2,
            help="Embedding vector dimension"
        )
    
    return params


async def run_benchmark_async(
    settings: Settings,
    params: Dict[str, Any]
) -> Dict[str, BenchmarkResult]:
    """
    Run benchmark asynchronously.
    
    Args:
        settings: Application settings
        params: Benchmark parameters
        
    Returns:
        Dictionary mapping db_name to BenchmarkResult
    """
    benchmark = VectorDBBenchmark(settings)
    
    results = await benchmark.run_benchmark(
        num_documents=params["num_documents"],
        dimension=params["dimension"],
        num_queries=params["num_queries"]
    )
    
    # Filter results based on selected databases
    filtered_results = {
        db_name: result
        for db_name, result in results.items()
        if db_name in params["databases"]
    }
    
    return filtered_results


def render_historical_results() -> None:
    """Render historical benchmark results."""
    if not st.session_state.benchmark_history:
        st.info("No historical benchmark results available")
        return
    
    st.subheader("📊 Historical Results")
    
    # Create timeline chart
    timeline_data = []
    
    for entry in st.session_state.benchmark_history:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        for db_name, result_data in entry["results"].items():
            timeline_data.append({
                'Timestamp': timestamp,
                'Database': db_name.capitalize(),
                'Throughput': result_data.get('throughput_ops_per_sec', 0),
                'p50 Latency': result_data.get('latency_p50_ms', 0)
            })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        # Throughput over time
        fig_throughput = px.line(
            df,
            x='Timestamp',
            y='Throughput',
            color='Database',
            title='Throughput Over Time',
            markers=True
        )
        st.plotly_chart(fig_throughput, use_container_width=True)
        
        # Latency over time
        fig_latency = px.line(
            df,
            x='Timestamp',
            y='p50 Latency',
            color='Database',
            title='p50 Latency Over Time',
            markers=True
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    
    # Show table of all results
    with st.expander("View All Historical Results"):
        for idx, entry in enumerate(reversed(st.session_state.benchmark_history)):
            st.markdown(f"**Run {len(st.session_state.benchmark_history) - idx}** - {entry['timestamp']}")
            st.json(entry["results"])


def main() -> None:
    """Main function for the Benchmark page."""
    st.set_page_config(
        page_title="Benchmark - Visual RAG",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load settings
    settings = Settings()
    
    # Render sidebar
    system_status = {
        "vector_db_connected": True,
        "vector_db_type": settings.default_vector_db,
        "llm_provider": settings.default_llm_provider,
        "llm_model": settings.default_model
    }
    
    selected_page = render_sidebar(
        current_page="benchmark",
        system_status=system_status,
        stats=None
    )
    
    # Handle page navigation
    if selected_page != "benchmark":
        st.switch_page(f"ui/pages/{selected_page}.py")
    
    # Main content
    st.title("📊 Vector Database Benchmark")
    st.markdown("Compare performance of Qdrant vs Milvus")
    
    st.divider()
    
    # Benchmark controls
    params = render_benchmark_controls()
    
    # Run benchmark button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_button = st.button(
            "🚀 Run Benchmark",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.benchmark_running or not params["databases"]
        )
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.benchmark_results = []
            st.rerun()
    
    with col3:
        if st.button("📜 Clear History", use_container_width=True):
            st.session_state.benchmark_history = []
            save_benchmark_history()
            st.rerun()
    
    # Run benchmark
    if run_button:
        st.session_state.benchmark_running = True
        
        with st.spinner("Running benchmark... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                status_text.text("Initializing benchmark...")
                progress_bar.progress(10)
                
                # Run benchmark
                status_text.text("Running benchmark tests...")
                progress_bar.progress(30)
                
                results = asyncio.run(run_benchmark_async(settings, params))
                
                progress_bar.progress(90)
                status_text.text("Processing results...")
                
                # Store results
                st.session_state.benchmark_results = results
                
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "params": params,
                    "results": {
                        db_name: result.model_dump(mode='json')
                        for db_name, result in results.items()
                    }
                }
                st.session_state.benchmark_history.append(history_entry)
                save_benchmark_history()
                
                progress_bar.progress(100)
                status_text.text("Benchmark complete!")
                
                st.success("✅ Benchmark completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Benchmark failed: {str(e)}")
                st.exception(e)
            
            finally:
                st.session_state.benchmark_running = False
    
    st.divider()
    
    # Display results
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        st.subheader("📈 Benchmark Results")
        
        # Summary metrics
        st.markdown("### Key Metrics")
        
        cols = st.columns(len(results))
        
        for idx, (db_name, result) in enumerate(results.items()):
            with cols[idx]:
                st.markdown(f"**{db_name.capitalize()}**")
                st.metric("Throughput", f"{result.throughput_ops_per_sec:.2f} ops/s")
                st.metric("p50 Latency", f"{result.latency_p50_ms:.2f} ms")
                st.metric("p95 Latency", f"{result.latency_p95_ms:.2f} ms")
                st.metric("Memory Usage", f"{result.memory_usage_mb:.2f} MB")
        
        st.divider()
        
        # Visualizations
        st.markdown("### Performance Visualizations")
        
        # Latency chart
        col1, col2 = st.columns(2)
        
        with col1:
            latency_chart = create_latency_chart(results)
            st.plotly_chart(latency_chart, use_container_width=True)
        
        with col2:
            throughput_chart = create_throughput_chart(results)
            st.plotly_chart(throughput_chart, use_container_width=True)
        
        # Recall and memory charts
        col1, col2 = st.columns(2)
        
        with col1:
            recall_chart = create_recall_chart(results)
            st.plotly_chart(recall_chart, use_container_width=True)
        
        with col2:
            memory_chart = create_memory_chart(results)
            st.plotly_chart(memory_chart, use_container_width=True)
        
        st.divider()
        
        # Detailed comparison table
        st.markdown("### Detailed Comparison")
        
        comparison_df = create_comparison_table(results)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Export options
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            json_data = json.dumps(
                {db_name: result.model_dump(mode='json') for db_name, result in results.items()},
                indent=2
            )
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export as CSV
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.divider()
    
    # Historical results
    render_historical_results()
    
    # Information section
    with st.expander("ℹ️ About Benchmarking"):
        st.markdown("""
        ### Benchmark Metrics
        
        **Latency Percentiles:**
        - **p50 (Median)**: 50% of queries complete faster than this time
        - **p95**: 95% of queries complete faster than this time
        - **p99**: 99% of queries complete faster than this time
        
        **Throughput:**
        - Number of operations (queries or indexing) per second
        - Higher is better
        
        **Recall@k:**
        - Percentage of relevant results found in top-k results
        - Measures search quality
        - Higher is better (1.0 = perfect recall)
        
        **Memory Usage:**
        - RAM consumed by the database during operations
        - Lower is better
        
        ### Interpretation
        
        - **Lower latency** = Faster query responses
        - **Higher throughput** = More queries handled per second
        - **Higher recall** = Better search quality
        - **Lower memory** = More efficient resource usage
        
        ### Tips
        
        - Run benchmarks multiple times for consistent results
        - Use realistic document counts for your use case
        - Consider both speed and quality metrics
        - Test with your actual embedding dimensions
        """)


if __name__ == "__main__":
    main()
