"""
RAG Chain — Retrieval-Augmented Generation pipeline.
Combines vector search with conversation context and LLM generation.
"""

import logging
from typing import Dict, List, Optional

from app.llm_provider import LLMProvider
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# System prompt template for RAG
RAG_SYSTEM_PROMPT = """You are an expert industrial maintenance AI assistant powered by a knowledge base of maintenance manuals, procedures, and equipment documentation.

RETRIEVED KNOWLEDGE (from maintenance documents):
{retrieved_context}

CONVERSATION HISTORY:
{conversation_history}

ENTITY CONTEXT (known facts about mentioned equipment):
{entity_context}

---

INSTRUCTIONS:
- Answer the user's question based primarily on the Retrieved Knowledge when relevant
- When citing information from documents, mention the source document name
- If the knowledge base doesn't contain relevant information, use your general industrial maintenance expertise but clearly state "Based on general knowledge:"
- Be specific, actionable, and practical
- For safety-critical advice, always recommend consulting qualified personnel
- Maintain context from the conversation history for follow-up questions

USER QUESTION: {user_query}"""


class RAGChain:
    """Retrieval-Augmented Generation chain combining ChromaDB retrieval with Gemini."""

    def __init__(self, vector_store: VectorStore, llm_provider: LLMProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    def _format_retrieved_docs(self, documents: List[Dict]) -> str:
        """Format retrieved documents into a prompt-friendly string."""
        if not documents:
            return "No relevant documents found in the knowledge base."

        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            distance = doc.get('distance', 0)
            relevance = f"{(1 - distance) * 100:.0f}%" if distance < 1 else "N/A"
            formatted.append(
                f"[Document {i}] Source: {source} | Relevance: {relevance}\n"
                f"{doc['content']}"
            )

        return "\n\n---\n\n".join(formatted)

    async def query(self, user_query: str,
                    conversation_history: str = "",
                    entity_context: str = "",
                    filters: Optional[Dict] = None,
                    n_results: int = 5) -> Dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks from ChromaDB
        2. Build augmented prompt with context + history
        3. Generate response with Gemini
        4. Return response + source documents
        """

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.query(
            query_text=user_query,
            n_results=n_results,
            where=filters
        )

        # Step 2: Format retrieved context
        retrieved_context = self._format_retrieved_docs(retrieved_docs)

        # Step 3: Build augmented prompt
        prompt = RAG_SYSTEM_PROMPT.format(
            retrieved_context=retrieved_context,
            conversation_history=conversation_history or "No previous conversation.",
            entity_context=entity_context or "No specific entity context.",
            user_query=user_query
        )

        # Step 4: Generate response
        response = await self.llm_provider.generate(prompt)

        # Step 5: Package results
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'source': doc.get('metadata', {}).get('source', 'Unknown'),
                'chunk_index': doc.get('metadata', {}).get('chunk_index', 0),
                'relevance': f"{(1 - doc.get('distance', 0)) * 100:.0f}%",
                'preview': doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
            })

        return {
            'response': response,
            'sources': sources,
            'num_sources': len(sources),
            'has_knowledge_base': len(retrieved_docs) > 0
        }

    async def simple_query(self, user_query: str) -> str:
        """Simple query without memory or entity context — for quick lookups."""
        result = await self.query(user_query)
        return result['response']
