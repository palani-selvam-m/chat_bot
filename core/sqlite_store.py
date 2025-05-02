import sqlite3
from typing import List, Dict, Any
import json
import time
from contextlib import contextmanager

class SQLiteStore:
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        print(f"\nInitializing SQLiteStore with database path: {self.db_path}")
        try:
            self._init_db()
        except Exception as e:
            print(f"Error during SQLiteStore initialization: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        print(f"\nAttempting to connect to database: {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Set journal mode to WAL for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            print("✓ Database connection established")
            try:
                yield conn
                # Explicitly commit any pending changes
                conn.commit()
                print("✓ Changes committed to database")
            except Exception as e:
                conn.rollback()
                print(f"! Rolling back changes due to error: {e}")
                raise
            finally:
                conn.close()
                print("✓ Database connection closed")
        except Exception as e:
            print(f"Error managing database connection: {e}")
            raise

    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        with self.get_connection() as conn:
            # Drop tables in correct order (child tables first)
            conn.execute("DROP TABLE IF EXISTS llm_traces")
            conn.execute("DROP TABLE IF EXISTS queries")
            conn.execute("DROP TABLE IF EXISTS chunks")
            
            # Create tables in correct order (parent tables first)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    chapter TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    translated_query TEXT,
                    search_time REAL,
                    ragas_metrics TEXT,
                    results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    execution_time REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
            """)
            
            conn.commit()

    def store_chunk(self, vector_id, chunk_data):
        """Store a chunk in the database"""
        print(f"\nStoring chunk {vector_id} in SQLite...")
        print(f"Chunk data keys: {chunk_data.keys()}")
        
        try:
            # Validate required fields
            required_fields = ["chunk_text", "source_file", "chapter"]
            missing_fields = [field for field in required_fields if field not in chunk_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            with self.get_connection() as conn:
                # Convert metadata to JSON string
                metadata_json = json.dumps(chunk_data.get("metadata", {}))
                
                # Print the values being inserted
                print(f"Inserting values:")
                print(f"vector_id: {vector_id}")
                print(f"chunk_text length: {len(chunk_data['chunk_text'])}")
                print(f"source_file: {chunk_data['source_file']}")
                print(f"chapter: {chunk_data['chapter']}")
                print(f"metadata: {metadata_json[:100]}...")  # Print first 100 chars of metadata
                
                # Insert chunk
                conn.execute("""
                    INSERT INTO chunks (vector_id, chunk_text, source_file, chapter, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    vector_id,
                    chunk_data["chunk_text"],
                    chunk_data["source_file"],
                    chunk_data["chapter"],
                    metadata_json
                ))
                
                # Explicitly commit the transaction
                conn.commit()
                print("✓ Transaction committed")
                
                # Verify the insertion
                cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE vector_id = ?", (vector_id,))
                count = cursor.fetchone()[0]
                if count == 1:
                    print(f"✓ Successfully stored and verified chunk {vector_id}")
                    return True
                else:
                    raise Exception(f"Failed to verify chunk {vector_id} insertion")
                
        except Exception as e:
            print(f"Error storing chunk {vector_id}: {str(e)}")
            print(f"Chunk data: {chunk_data}")
            raise

    def verify_chunk_storage(self, vector_id):
        """Verify if a chunk exists in the database"""
        print(f"\nVerifying chunk {vector_id} in database...")
        try:
            with self.get_connection() as conn:
                # First check if the table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='chunks'
                """)
                if not cursor.fetchone():
                    print("! Chunks table does not exist")
                    return None
                
                # Then check for the chunk
                cursor = conn.execute("""
                    SELECT vector_id, chunk_text, source_file, chapter, metadata
                    FROM chunks
                    WHERE vector_id = ?
                """, (vector_id,))
                
                row = cursor.fetchone()
                if row:
                    print(f"✓ Found chunk {vector_id} in database")
                    return {
                        "vector_id": row[0],
                        "chunk_text": row[1],
                        "source_file": row[2],
                        "chapter": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {}
                    }
                else:
                    print(f"✗ Chunk {vector_id} not found in database")
                    return None
        except Exception as e:
            print(f"Error verifying chunk {vector_id}: {e}")
            return None

    def get_chunks_by_ids(self, vector_ids):
        """Retrieve chunks by their vector IDs"""
        print(f"\nRetrieving chunks from SQLite: {vector_ids}")
        try:
            with self.get_connection() as conn:
                # Create placeholders for the IN clause
                placeholders = ','.join('?' * len(vector_ids))
                
                # Query chunks
                cursor = conn.execute(f"""
                    SELECT vector_id, chunk_text, source_file, chapter, metadata
                    FROM chunks
                    WHERE vector_id IN ({placeholders})
                """, vector_ids)
                
                chunks = []
                for row in cursor.fetchall():
                    chunk = {
                        "vector_id": row[0],
                        "chunk_text": row[1],
                        "source_file": row[2],
                        "chapter": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {}
                    }
                    chunks.append(chunk)
                
                print(f"✓ Retrieved {len(chunks)} chunks")
                return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def log_query(self, query_text: str, results: List[Dict[str, Any]], search_time: float, 
                 translated_query: str = None, ragas_metrics: Dict[str, float] = None):
        """Log a query and its results to the database"""
        print("\nLogging query to database...")
        try:
            with self.get_connection() as conn:
                # Convert results and metrics to JSON strings
                results_json = json.dumps(results) if results else None
                metrics_json = json.dumps(ragas_metrics) if ragas_metrics else None
                
                # Insert query record
                cursor = conn.execute("""
                    INSERT INTO queries (
                        query_text, translated_query, search_time, ragas_metrics, results
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    query_text,
                    translated_query,
                    search_time,
                    metrics_json,
                    results_json
                ))
                
                # Get the inserted query ID
                query_id = cursor.lastrowid
                
                # Commit the transaction
                conn.commit()
                print(f"✓ Successfully logged query {query_id}")
                
                return query_id
        except Exception as e:
            print(f"Error logging query: {e}")
            raise

    def log_llm_trace(self, query_id: int, prompt: str, response: str, 
                     execution_time: float, metadata: Dict[str, Any] = None):
        """Log an LLM trace to the database"""
        print(f"\nLogging LLM trace for query {query_id}...")
        try:
            with self.get_connection() as conn:
                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata) if metadata else None
                
                # Insert trace record
                cursor = conn.execute("""
                    INSERT INTO llm_traces (
                        query_id, prompt, response, execution_time, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    query_id,
                    prompt,
                    response,
                    execution_time,
                    metadata_json
                ))
                
                # Get the inserted trace ID
                trace_id = cursor.lastrowid
                
                # Commit the transaction
                conn.commit()
                print(f"✓ Successfully logged trace {trace_id}")
                
                return trace_id
        except Exception as e:
            print(f"Error logging LLM trace: {e}")
            raise

    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT q.*, 
                       (SELECT COUNT(*) FROM llm_traces WHERE query_id = q.id) as trace_count
                FROM queries q
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_llm_traces(self, query_id: int) -> List[Dict[str, Any]]:
        """Retrieve LLM traces for a specific query"""
        print(f"\nRetrieving LLM traces for query {query_id}...")
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT id, query_id, prompt, response, execution_time, metadata, created_at
                    FROM llm_traces
                    WHERE query_id = ?
                    ORDER BY created_at
                """, (query_id,))
                
                traces = []
                for row in cursor.fetchall():
                    trace = {
                        "id": row["id"],
                        "query_id": row["query_id"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "execution_time": row["execution_time"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "created_at": row["created_at"]
                    }
                    traces.append(trace)
                
                print(f"✓ Retrieved {len(traces)} traces")
                return traces
        except Exception as e:
            print(f"Error retrieving LLM traces: {e}")
            return []

    def get_database_stats(self):
        """Get statistics about the database state"""
        print("\nChecking database state...")
        try:
            with self.get_connection() as conn:
                # Get chunk count
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]
                
                # Get query count
                cursor = conn.execute("SELECT COUNT(*) FROM queries")
                query_count = cursor.fetchone()[0]
                
                # Get trace count
                cursor = conn.execute("SELECT COUNT(*) FROM llm_traces")
                trace_count = cursor.fetchone()[0]
                
                # Get chunk size statistics
                cursor = conn.execute("""
                    SELECT 
                        MIN(LENGTH(chunk_text)) as min_size,
                        MAX(LENGTH(chunk_text)) as max_size,
                        AVG(LENGTH(chunk_text)) as avg_size
                    FROM chunks
                """)
                size_stats = cursor.fetchone()
                
                print("\nDatabase Statistics:")
                print(f"Total chunks: {chunk_count}")
                print(f"Total queries: {query_count}")
                print(f"Total traces: {trace_count}")
                if size_stats:
                    print("\nChunk Size Statistics:")
                    print(f"Min size: {size_stats[0]} characters")
                    print(f"Max size: {size_stats[1]} characters")
                    print(f"Avg size: {size_stats[2]:.0f} characters")
                
                return {
                    "chunk_count": chunk_count,
                    "query_count": query_count,
                    "trace_count": trace_count,
                    "size_stats": {
                        "min": size_stats[0],
                        "max": size_stats[1],
                        "avg": size_stats[2]
                    } if size_stats else None
                }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return None

    def clear_database(self):
        """Clear all data from the database"""
        print("\nClearing SQLite database...")
        try:
            with self.get_connection() as conn:
                # Drop all tables
                conn.execute("DROP TABLE IF EXISTS llm_traces")
                conn.execute("DROP TABLE IF EXISTS queries")
                conn.execute("DROP TABLE IF EXISTS chunks")
                
                # Recreate tables
                self._init_db()
                
                print("✓ Database cleared successfully")
                return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            raise 