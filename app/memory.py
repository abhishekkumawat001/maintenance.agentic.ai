"""
Memory System — Conversation memory with auto-summarization and entity tracking.
Provides contextual awareness across chat turns and sessions.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from app.config import MemoryConfig
from app.knowledge_base import MaintenanceKnowledgeBase
from app.llm_provider import LLMProvider
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Sliding-window conversation memory with auto-summarization.

    How it works:
    - Keeps the last N messages in full detail (short-term)
    - When the window overflows, older messages are summarized by Gemini
    - The summary + recent messages form the context for each new prompt
    - All messages are persisted to SQLite for cross-session recall
    """

    def __init__(self, knowledge_base: MaintenanceKnowledgeBase,
                 session_id: str = None):
        self.knowledge_base = knowledge_base
        self.session_id = session_id or str(uuid4())
        self.messages: List[Dict] = []  # {role, content, timestamp, sources}
        self.summary: str = ""
        self.max_turns = MemoryConfig.MAX_TURNS
        self.summary_threshold = MemoryConfig.SUMMARY_THRESHOLD
        self.max_context_chars = MemoryConfig.MAX_CONTEXT_CHARS

        # Load existing session if resuming
        self._load_from_db()

    def _load_from_db(self):
        """Load existing conversation from database."""
        history = self.knowledge_base.get_conversation_history(self.session_id)
        if history:
            self.messages = [
                {
                    'role': msg['role'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'],
                    'sources': json.loads(msg['sources']) if msg['sources'] else []
                }
                for msg in history
            ]
            logger.info("Loaded %d messages for session %s", len(self.messages), self.session_id)

        # Load existing summary
        summary_data = self.knowledge_base.get_conversation_summary(self.session_id)
        if summary_data:
            self.summary = summary_data['summary']

    def add_message(self, role: str, content: str, sources: list = None):
        """Add a message to memory and persist to database."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or []
        }
        self.messages.append(message)

        # Persist to SQLite
        self.knowledge_base.save_message(
            session_id=self.session_id,
            role=role,
            content=content,
            sources=sources
        )

    async def auto_summarize(self, llm_provider: LLMProvider):
        """
        When conversation exceeds threshold, summarize older messages
        to keep the context window manageable.
        """
        if len(self.messages) < self.summary_threshold:
            return

        # Take the oldest messages (beyond what we keep in full)
        keep_recent = self.max_turns // 2
        messages_to_summarize = self.messages[:-keep_recent]

        if not messages_to_summarize:
            return

        # Format messages for summarization
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
        )

        prompt = f"""Summarize this maintenance conversation concisely. 
Preserve key facts: equipment mentioned, problems discussed, solutions suggested, and any decisions made.

Previous summary: {self.summary or 'None'}

New messages to summarize:
{conversation_text}

Provide a concise summary (max 300 words) that captures all important context."""

        new_summary = await llm_provider.generate(prompt)

        # Update summary
        self.summary = new_summary

        # Keep only recent messages in the active list
        self.messages = self.messages[-keep_recent:]

        # Persist summary
        self.knowledge_base.save_conversation_summary(
            session_id=self.session_id,
            summary=self.summary,
            message_count=len(self.messages)
        )

        logger.info("Auto-summarized conversation %s", self.session_id)

    def get_context_string(self) -> str:
        """
        Build the context string for prompt injection.
        Format: [Summary of older messages] + [Recent N messages]
        """
        parts = []

        # Add summary of older conversation if exists
        if self.summary:
            parts.append(f"[Previous conversation summary]\n{self.summary}")

        # Add recent messages
        recent = self.messages[-self.max_turns:]
        if recent:
            formatted_messages = []
            for msg in recent:
                role_label = "User" if msg['role'] == 'user' else "Assistant"
                formatted_messages.append(f"{role_label}: {msg['content']}")
            parts.append("\n".join(formatted_messages))

        context = "\n\n".join(parts)

        # Truncate if too long
        if len(context) > self.max_context_chars:
            context = context[-self.max_context_chars:]

        return context

    def get_message_count(self) -> int:
        """Total messages in this session (including summarized ones)."""
        return len(self.messages)

    def clear(self):
        """Clear current session memory."""
        self.messages = []
        self.summary = ""


class EntityMemory:
    """
    Tracks facts about equipment entities across conversations.
    Uses the vector store's entity collection for semantic retrieval.
    """

    def __init__(self, vector_store: VectorStore, llm_provider: LLMProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    async def extract_and_store_entities(self, user_message: str,
                                          assistant_response: str):
        """
        Extract equipment-related entities and facts from a conversation turn,
        then store them in the vector store for future retrieval.
        """
        prompt = f"""Extract equipment-related facts from this conversation turn.
For each piece of equipment or component mentioned, extract key facts.

USER: {user_message}
ASSISTANT: {assistant_response}

If equipment or specific maintenance facts are mentioned, output JSON array:
[{{"entity": "equipment name/ID", "facts": "key facts about this entity"}}]

If no specific equipment facts are mentioned, output: []"""

        try:
            response = await self.llm_provider.generate(prompt)
            # Try to parse JSON from response
            # Handle markdown code blocks
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("\n", 1)[-1]
                clean_response = clean_response.rsplit("```", 1)[0]

            entities = json.loads(clean_response)

            for entity_data in entities:
                entity_name = entity_data.get('entity', '')
                facts = entity_data.get('facts', '')
                if entity_name and facts:
                    entity_id = f"entity_{entity_name.lower().replace(' ', '_')}"
                    self.vector_store.add_entity(
                        entity_id=entity_id,
                        text=f"{entity_name}: {facts}",
                        metadata={
                            'entity_name': entity_name,
                            'updated_at': datetime.now().isoformat()
                        }
                    )
                    logger.info("Stored entity: %s", entity_name)

        except (json.JSONDecodeError, Exception) as e:
            # Entity extraction is best-effort — don't break the flow
            logger.debug("Entity extraction skipped: %s", e)

    def get_entity_context(self, query: str) -> str:
        """Retrieve relevant entity facts for a given query."""
        entities = self.vector_store.query_entities(query, n_results=3)
        if not entities:
            return ""

        context_parts = []
        for entity in entities:
            context_parts.append(f"• {entity['content']}")

        return "\n".join(context_parts)
