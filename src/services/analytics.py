"""
Analytics service for tracking usage and events.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from ..config.settings import get_settings
from ..models.schemas import AnalyticsEvent

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalyticsService:
    """Service for tracking and analyzing application usage."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.analytics_db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for analytics."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                word_count INTEGER,
                processed_at TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                response_length INTEGER,
                sources_count INTEGER,
                timestamp TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Analytics database initialized at {self.db_path}")

    def track_event(self, event: AnalyticsEvent) -> None:
        """
        Track an analytics event.

        Args:
            event: AnalyticsEvent to track.
        """
        if not settings.enable_analytics:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO events (event_type, session_id, timestamp, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            event.event_type,
            event.session_id,
            event.timestamp.isoformat(),
            json.dumps(event.metadata)
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Tracked event: {event.event_type}")

    def track_document_processed(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        word_count: int
    ) -> None:
        """Track document processing."""
        if not settings.enable_analytics:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO documents (session_id, filename, file_type, word_count, processed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, filename, file_type, word_count, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def track_query(
        self,
        session_id: str,
        question: str,
        response_length: int,
        sources_count: int
    ) -> None:
        """Track a user query."""
        if not settings.enable_analytics:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO queries (session_id, question, response_length, sources_count, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, question, response_length, sources_count, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Document count
        cursor.execute(
            'SELECT COUNT(*) FROM documents WHERE session_id = ?',
            (session_id,)
        )
        doc_count = cursor.fetchone()[0]

        # Query count
        cursor.execute(
            'SELECT COUNT(*) FROM queries WHERE session_id = ?',
            (session_id,)
        )
        query_count = cursor.fetchone()[0]

        # Total words processed
        cursor.execute(
            'SELECT SUM(word_count) FROM documents WHERE session_id = ?',
            (session_id,)
        )
        total_words = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'documents_processed': doc_count,
            'queries_made': query_count,
            'total_words_processed': total_words
        }

    def get_usage_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for the past N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total events
        cursor.execute('SELECT COUNT(*) FROM events')
        total_events = cursor.fetchone()[0]

        # Total documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]

        # Total queries
        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]

        # Popular file types
        cursor.execute('''
            SELECT file_type, COUNT(*) as count
            FROM documents
            GROUP BY file_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        file_types = dict(cursor.fetchall())

        conn.close()

        return {
            'total_events': total_events,
            'total_documents': total_docs,
            'total_queries': total_queries,
            'popular_file_types': file_types
        }

    def export_analytics(self, output_path: str) -> None:
        """Export analytics data to JSON."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = {
            'exported_at': datetime.now().isoformat(),
            'events': [],
            'documents': [],
            'queries': []
        }

        cursor.execute('SELECT * FROM events')
        for row in cursor.fetchall():
            data['events'].append({
                'id': row[0],
                'event_type': row[1],
                'session_id': row[2],
                'timestamp': row[3],
                'metadata': json.loads(row[4]) if row[4] else {}
            })

        cursor.execute('SELECT * FROM documents')
        for row in cursor.fetchall():
            data['documents'].append({
                'id': row[0],
                'session_id': row[1],
                'filename': row[2],
                'file_type': row[3],
                'word_count': row[4],
                'processed_at': row[5]
            })

        cursor.execute('SELECT * FROM queries')
        for row in cursor.fetchall():
            data['queries'].append({
                'id': row[0],
                'session_id': row[1],
                'question': row[2],
                'response_length': row[3],
                'sources_count': row[4],
                'timestamp': row[5]
            })

        conn.close()

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analytics exported to {output_path}")
