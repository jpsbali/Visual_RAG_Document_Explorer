"""
Memory Load Node - Load conversation history and relevant past context.

This node loads both short-term (session) and long-term (vector DB) memory
to provide conversational context for the current query.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def memory_load_node(state: AgentState) -> dict[str, Any]:
    """
    Load conversational memory from session state and vector DB.
    
    Steps:
    1. Load short-term memory from session state (last N messages)
    2. Search vector DB for relevant past conversations
    3. Calculate context window usage with tiktoken
    4. Trigger summarization if usage > 60%
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with memory fields
    """
    try:
        # Import dependencies
        from config.settings import Settings
        import tiktoken
        
        settings = state.get("_settings") or Settings()
        
        # 1. Load short-term memory from session state
        # In production, this would come from st.session_state.chat_history
        chat_history = state.get("chat_history", [])
        
        # 2. Search vector DB for relevant past conversations
        relevant_memory = []
        
        if settings.enable_long_term_memory:
            try:
                from core.vectordb.router import get_vector_store
                from core.embeddings.embedding_router import get_embedding_service
                
                vector_store = get_vector_store(settings)
                embedding_service = get_embedding_service(settings)
                
                # Embed current query for similarity search
                query_embedding = await embedding_service.embed_query(state["query"])
                
                # Search for similar past conversations
                memory_results = await vector_store.search(
                    collection="conversation_memory",
                    query_embedding=query_embedding,
                    top_k=3,
                    filters={"session_id": state["session_id"]} if state.get("session_id") else None
                )
                
                relevant_memory = [
                    {
                        "query": r.metadata.get("query", ""),
                        "answer": r.metadata.get("answer", ""),
                        "timestamp": r.metadata.get("timestamp", 0)
                    }
                    for r in memory_results
                ]
            except Exception as e:
                # Memory collection may not exist yet or vector DB unavailable
                logger.warning(f"Failed to load long-term memory: {e}")
                relevant_memory = []
        
        # 3. Calculate context window usage
        try:
            encoding = tiktoken.encoding_for_model(settings.default_model)
        except:
            # Fallback to cl100k_base if model not found
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Count tokens in chat history
        history_tokens = sum(
            len(encoding.encode(msg.get("content", "")))
            for msg in chat_history
        )
        
        # Count tokens in relevant memory
        memory_tokens = sum(
            len(encoding.encode(f"{m['query']} {m['answer']}"))
            for m in relevant_memory
        )
        
        # Model context window (default 128k)
        context_window = settings.llm_context_window
        
        # Reserve 40% for retrieval + generation
        available_for_history = context_window * 0.6
        
        context_usage = (history_tokens + memory_tokens) / available_for_history if available_for_history > 0 else 0.0
        
        # 4. Trigger summarization if needed
        memory_summary = None
        if context_usage > settings.context_window_threshold and len(chat_history) > 10:
            try:
                from core.llm.llm_router import get_llm_provider
                
                llm = get_llm_provider(settings)
                
                # Summarize older messages (all but last 5)
                old_messages = chat_history[:-5]
                messages_text = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in old_messages
                )
                
                system_prompt = "Summarize the following conversation history concisely, preserving key context and information."
                memory_summary = await llm.generate(
                    prompt=messages_text,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=500
                )
                
                # Keep only recent messages + summary
                chat_history = [
                    {"role": "system", "content": f"Previous conversation summary: {memory_summary}"}
                ] + chat_history[-5:]
                
                logger.info(f"Summarized {len(old_messages)} messages due to context window usage: {context_usage:.2%}")
            except Exception as e:
                logger.warning(f"Failed to summarize conversation history: {e}")
                # Keep full history if summarization fails
        
        return {
            "chat_history": chat_history,
            "relevant_memory": relevant_memory,
            "context_window_usage": context_usage,
            "memory_summary": memory_summary
        }
    
    except Exception as e:
        logger.error(f"Error in memory_load_node: {e}")
        # Return minimal state on error
        return {
            "chat_history": state.get("chat_history", []),
            "relevant_memory": [],
            "context_window_usage": 0.0,
            "memory_summary": None
        }
