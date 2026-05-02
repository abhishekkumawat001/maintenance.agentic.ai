"""
Knowledge Base — SQLite storage for equipment, maintenance history,
fault patterns, and conversation history.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

from app.config import DatabaseConfig

logger = logging.getLogger(__name__)


class MaintenanceKnowledgeBase:
    """SQLite-backed knowledge base with connection pooling via context manager."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DatabaseConfig.DB_PATH
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for safe SQLite connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize all database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Equipment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equipment (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    model TEXT,
                    manufacturer TEXT,
                    installation_date DATE,
                    last_maintenance DATE
                )
            ''')

            # Maintenance history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maintenance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_id TEXT,
                    maintenance_date DATE,
                    type TEXT,
                    description TEXT,
                    technician TEXT,
                    cost REAL,
                    downtime_hours INTEGER,
                    FOREIGN KEY (equipment_id) REFERENCES equipment (id)
                )
            ''')

            # Fault patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fault_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_type TEXT,
                    symptoms TEXT,
                    root_cause TEXT,
                    solution TEXT,
                    confidence_score REAL
                )
            ''')

            # Conversation history table (NEW)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Conversation summaries table (NEW)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    key_entities TEXT DEFAULT '[]',
                    message_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Ingested documents tracking table (NEW)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    file_type TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    # ─── Equipment Methods ───

    def add_equipment(self, equipment_data: Dict) -> bool:
        """Add or update equipment in database."""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO equipment
                    (id, name, type, model, manufacturer, installation_date, last_maintenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    equipment_data['id'],
                    equipment_data['name'],
                    equipment_data['type'],
                    equipment_data.get('model', 'Unknown'),
                    equipment_data.get('manufacturer', 'Unknown'),
                    equipment_data.get('installation_date', datetime.now().strftime('%Y-%m-%d')),
                    equipment_data.get('last_maintenance', 'Never')
                ))
            return True
        except Exception as e:
            logger.error("Error adding equipment: %s", e)
            return False

    def get_all_equipment(self) -> List[Dict]:
        """Get all equipment from database."""
        with self._get_connection() as conn:
            rows = conn.execute('SELECT * FROM equipment').fetchall()
            return [dict(row) for row in rows]

    def get_maintenance_history(self, equipment_id: str) -> List[Dict]:
        """Retrieve maintenance history for specific equipment."""
        with self._get_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM maintenance_history WHERE equipment_id = ? ORDER BY maintenance_date DESC',
                (equipment_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def find_similar_faults(self, symptoms: str) -> List[Dict]:
        """Find similar fault patterns using keyword matching."""
        with self._get_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM fault_patterns WHERE symptoms LIKE ? ORDER BY confidence_score DESC',
                (f'%{symptoms}%',)
            ).fetchall()
            return [dict(row) for row in rows]

    # ─── Conversation Methods (NEW) ───

    def save_message(self, session_id: str, role: str, content: str, sources: list = None):
        """Save a chat message to the database."""
        with self._get_connection() as conn:
            conn.execute(
                'INSERT INTO conversations (session_id, role, content, sources) VALUES (?, ?, ?, ?)',
                (session_id, role, content, json.dumps(sources or []))
            )

    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve conversation history for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                'SELECT role, content, sources, timestamp FROM conversations '
                'WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?',
                (session_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def save_conversation_summary(self, session_id: str, summary: str,
                                   key_entities: list = None, message_count: int = 0):
        """Save or update a conversation summary."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO conversation_summaries
                (session_id, summary, key_entities, message_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (session_id, summary, json.dumps(key_entities or []), message_count))

    def get_conversation_summary(self, session_id: str) -> Optional[Dict]:
        """Get the summary for a conversation session."""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM conversation_summaries WHERE session_id = ?',
                (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_sessions(self) -> List[Dict]:
        """Get all conversation sessions with metadata."""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT session_id, MIN(timestamp) as started_at,
                       MAX(timestamp) as last_message, COUNT(*) as message_count
                FROM conversations GROUP BY session_id
                ORDER BY last_message DESC
            ''').fetchall()
            return [dict(row) for row in rows]

    # ─── Document Tracking Methods (NEW) ───

    def track_document(self, filename: str, file_path: str,
                       file_type: str, chunk_count: int) -> int:
        """Track an ingested document. Returns the document ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                'INSERT INTO ingested_documents (filename, file_path, file_type, chunk_count) '
                'VALUES (?, ?, ?, ?)',
                (filename, file_path, file_type, chunk_count)
            )
            return cursor.lastrowid

    def get_ingested_documents(self) -> List[Dict]:
        """Get all ingested documents."""
        with self._get_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM ingested_documents ORDER BY ingested_at DESC'
            ).fetchall()
            return [dict(row) for row in rows]

    def is_document_ingested(self, filename: str) -> bool:
        """Check if a document has already been ingested."""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT id FROM ingested_documents WHERE filename = ?',
                (filename,)
            ).fetchone()
            return row is not None
