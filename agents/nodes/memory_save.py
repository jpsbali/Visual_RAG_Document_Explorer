"""
Memory Save Node - Save conversation to memory.

This node saves the current Q&A exchange to both short-term (session) and
long-term (vector DB) memory for future retrieval.
"""

from typing import Any
from agents.state import AgentState
import logging
import time

logger = logging.getLogger(__name__)


async def memory_save_node(state: AgentState) -> dict[str, Any]:
    """
    Save conversation to memory.
    
    Steps:
    1. Add Q&A to chat_history
    2. Store in vector DB for long-term memory
    3. Summarize if context window exceeded
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with memory updates
    """
    try:
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        
        # 1. Add to chat history
        chat_history = state.get("chat_history", []).copy()
        
        # Add user query
        chat_history.append({
            "role": "user",
            "content": state["query"],
            "timestamp": time.time()
        })
        
        # Add assistant answer
        chat_history.append({
            "role": "assistant",
            "content": state.get("final_answer", ""),
            "timestamp": time.time()
        })
        
        # Limit to short_term_memory_size
        if len(chat_history) > settings.short_term_memory_size:
            chat_history = chat_history[-settings.short_term_memory_size:]
        
        # 2. Store in vector DB for long-term memory
        if settings.enable_long_term_memory:
            try:
                from core.vectordb.router import get_vector_store
                from core.embeddings.embedding_router import get_embedding_service
                
                vector_store = get_vector_store(settings)
                embedding_service = get_embedding_service(settings)
                
                # Create conversation summary for embedding
                conversation_text = f"Q: {state['query']}\nA: {state.get('final_answer', '')}"
                embedding = await embedding_service.embed_query(conversation_text)
                
                # Generate unique ID
                conversation_id = f"{state.get('session_id', 'default')}_{int(time.time())}"
                
                # Store in conversation_memory collection
                await vector_store.upsert(
                    collection="conversation_memory",
                    ids=[conversation_id],
                    embeddings=[embedding],
                    documents=[conversation_text],
                    metadatas=[{
                        "session_id": state.get("session_id", "default"),
                        "query": state["query"],
                        "answer": state.get("final_answer", ""),
                        "timestamp": time.time()
                    }]
                )
                
                logger.info(f"Saved conversation to long-term memory: {conversation_id}")
            except Exception as e:
                # Memory storage is non-critical, log and continue
                logger.warning(f"Failed to save to long-term memory: {e}")
        
        # 3. Check if summarization needed
        memory_summary = state.get("memory_summary")
        context_usage = state.get("context_window_usage", 0.0)
        
        if context_usage > settings.context_window_threshold and len(chat_history) > 10:
            try:
                from core.llm.llm_router import get_llm_provider
                
                llm = get_llm_provider(settings)
                
                # Summarize older messages (all but last 5)
                old_messages = chat_history[:-5]
                if len(old_messages) > 5:
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
                        {"role": "system", "content": f"Previous conversation: {memory_summary}"}
                    ] + chat_history[-5:]
                    
                    logger.info(f"Summarized conversation history due to context window usage")
            except Exception as e:
                logger.warning(f"Failed to summarize conversation: {e}")
        
        return {
            "chat_history": chat_history,
            "memory_summary": memory_summary
        }
    
    except Exception as e:
        logger.error(f"Error in memory_save_node: {e}")
        # Return current state on error
        return {
            "chat_history": state.get("chat_history", []),
            "memory_summary": state.get("memory_summary")
        }
