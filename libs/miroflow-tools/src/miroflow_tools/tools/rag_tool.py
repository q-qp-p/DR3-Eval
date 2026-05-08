# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RAG Tool for Long Context Document Search

This module provides RAG (Retrieval-Augmented Generation) functionality
for searching and retrieving relevant passages from long context documents.

Features:
- Chunk-based document processing for handling long documents
- SQLite-based embedding cache for fast subsequent queries
- Semantic search using OpenAI embeddings
- Support for long_context.json format
- Batch embedding generation with progress tracking
"""

import json
import hashlib
import sqlite3
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# Try to import tiktoken for token-based chunking
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based chunking")


@dataclass
class ChunkInfo:
    """Information about a document chunk."""
    doc_index: int
    chunk_index: int
    title: str
    content: str
    url: str
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    """A single search result from RAG."""
    title: str
    content: str
    score: float
    doc_index: int
    chunk_index: int
    metadata: Dict[str, Any]


class RAGTool:
    """
    RAG Tool for semantic search over long context documents.
    
    Uses chunk-based processing to handle long documents and
    SQLite to cache embeddings for fast subsequent queries.
    
    Supports two chunking modes:
    - Token-based (recommended): Consistent information density across languages
    - Character-based (fallback): When tiktoken is not available
    """
    
    # Default chunk configuration
    DEFAULT_CHUNK_SIZE = 1500  # Characters per chunk (for char mode)
    DEFAULT_CHUNK_OVERLAP = 200  # Overlap between chunks (for char mode)
    DEFAULT_CHUNK_TOKENS = 500  # Tokens per chunk (for token mode, ~2000 chars for English, ~500 chars for Chinese)
    DEFAULT_CHUNK_OVERLAP_TOKENS = 50  # Overlap in tokens
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        use_token_chunking: bool = True,
        chunk_tokens: int = None,
        chunk_overlap_tokens: int = None
    ):
        """
        Initialize RAG Tool.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            embedding_model: Model to use for embeddings
            cache_dir: Directory to store SQLite cache (default: same as document)
            chunk_size: Maximum characters per chunk (default: 1500, used when use_token_chunking=False)
            chunk_overlap: Overlap between chunks in characters (default: 200)
            use_token_chunking: If True, use token-based chunking for consistent info density (default: True)
            chunk_tokens: Maximum tokens per chunk (default: 500, used when use_token_chunking=True)
            chunk_overlap_tokens: Overlap between chunks in tokens (default: 50)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        
        # Token-based chunking settings
        self.use_token_chunking = use_token_chunking and TIKTOKEN_AVAILABLE
        self.chunk_tokens = chunk_tokens or self.DEFAULT_CHUNK_TOKENS
        self.chunk_overlap_tokens = chunk_overlap_tokens or self.DEFAULT_CHUNK_OVERLAP_TOKENS
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if self.use_token_chunking:
            try:
                # Use cl100k_base encoding (used by text-embedding-3-small/large)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Using token-based chunking: {self.chunk_tokens} tokens per chunk")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}, falling back to character-based chunking")
                self.use_token_chunking = False
        
        if not self.use_token_chunking:
            logger.info(f"Using character-based chunking: {self.chunk_size} chars per chunk")
        
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[ChunkInfo] = []
        self.embeddings: np.ndarray = None
        self.current_file: str = None
        self.db_path: str = None
        self.conn: sqlite3.Connection = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for cache validation."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_db_path(self, file_path: str) -> str:
        """Get SQLite database path for caching embeddings."""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            return os.path.join(self.cache_dir, f"{base_name}.chunks.db")
        else:
            return file_path + ".chunks.db"
    
    def _init_db(self, db_path: str, readonly: bool = False):
        """
        Initialize SQLite database for embedding cache.
        
        Args:
            db_path: Path to the SQLite database file
            readonly: If True, open in read-only mode (for pre-built db files).
                     This allows multiple processes to read the same db file
                     simultaneously without conflicts.
        """
        if readonly:
            # Open in read-only mode using URI
            # This allows multiple processes to read the same file concurrently
            db_uri = f"file:{db_path}?mode=ro"
            self.conn = sqlite3.connect(db_uri, uri=True)
            logger.info(f"Opened database in READ-ONLY mode: {db_path}")
            # In read-only mode, tables should already exist, no need to create
            return
        else:
            self.conn = sqlite3.connect(db_path)
        
        cursor = self.conn.cursor()
        
        # Create tables (only in read-write mode)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_index INTEGER,
                chunk_index INTEGER,
                title TEXT,
                content TEXT,
                url TEXT,
                start_char INTEGER,
                end_char INTEGER,
                embedding BLOB
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_chunk 
            ON chunks(doc_index, chunk_index)
        ''')
        
        self.conn.commit()
    
    def _check_cache_valid(self, file_hash: str) -> bool:
        """Check if cache is valid for the current file."""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'file_hash'")
        result = cursor.fetchone()
        
        if result and result[0] == file_hash:
            cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_model'")
            model_result = cursor.fetchone()
            if model_result and model_result[0] == self.embedding_model:
                # Also check chunk configuration
                cursor.execute("SELECT value FROM metadata WHERE key = 'chunk_size'")
                chunk_size_result = cursor.fetchone()
                cursor.execute("SELECT value FROM metadata WHERE key = 'chunk_overlap'")
                chunk_overlap_result = cursor.fetchone()
                if (chunk_size_result and int(chunk_size_result[0]) == self.chunk_size and
                    chunk_overlap_result and int(chunk_overlap_result[0]) == self.chunk_overlap):
                    return True
        
        return False
    
    def _save_cache_metadata(self, file_hash: str):
        """Save cache metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('file_hash', file_hash)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('embedding_model', self.embedding_model)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('chunk_size', str(self.chunk_size))
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('chunk_overlap', str(self.chunk_overlap))
        )
        self.conn.commit()
    
    def _split_into_chunks_by_tokens(self, text: str, doc_index: int, title: str, url: str) -> List[ChunkInfo]:
        """
        Split a document into chunks based on token count.
        This ensures consistent information density across different languages.
        
        Args:
            text: The document text to split
            doc_index: Index of the source document
            title: Document title
            url: Document URL
            
        Returns:
            List of ChunkInfo objects
        """
        if not text or not text.strip():
            return [ChunkInfo(
                doc_index=doc_index,
                chunk_index=0,
                title=title,
                content="[Empty Document]",
                url=url,
                start_char=0,
                end_char=0
            )]
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_tokens:
            # Text is short enough, return as single chunk
            return [ChunkInfo(
                doc_index=doc_index,
                chunk_index=0,
                title=title,
                content=text.strip(),
                url=url,
                start_char=0,
                end_char=len(text)
            )]
        
        chunks = []
        chunk_index = 0
        token_start = 0
        
        while token_start < total_tokens:
            token_end = min(token_start + self.chunk_tokens, total_tokens)
            
            # Get the chunk tokens and decode back to text
            chunk_tokens = tokens[token_start:token_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Try to find a good break point at sentence boundary
            if token_end < total_tokens:
                # Look for sentence endings in the last 20% of the chunk
                search_start = int(len(chunk_text) * 0.8)
                best_break = -1
                for sep in ['。', '！', '？', '. ', '! ', '? ', '\n\n', '\n']:
                    break_pos = chunk_text.rfind(sep, search_start)
                    if break_pos > 0:
                        best_break = break_pos + len(sep)
                        break
                
                if best_break > 0:
                    chunk_text = chunk_text[:best_break]
                    # Recalculate token_end based on the actual text
                    actual_tokens = self.tokenizer.encode(chunk_text)
                    token_end = token_start + len(actual_tokens)
            
            chunk_text = chunk_text.strip()
            if chunk_text:
                # Calculate approximate character positions
                char_start = len(self.tokenizer.decode(tokens[:token_start]))
                char_end = len(self.tokenizer.decode(tokens[:token_end]))
                
                chunks.append(ChunkInfo(
                    doc_index=doc_index,
                    chunk_index=chunk_index,
                    title=title,
                    content=chunk_text,
                    url=url,
                    start_char=char_start,
                    end_char=char_end
                ))
                chunk_index += 1
            
            # Move to next chunk with overlap
            # If we've reached the end, break
            if token_end >= total_tokens:
                break
            
            # Calculate next start position with overlap
            next_start = token_end - self.chunk_overlap_tokens
            
            # Ensure we make progress (avoid infinite loop)
            # If next_start hasn't moved forward enough, just move to token_end
            if next_start <= token_start:
                next_start = token_end
            
            token_start = next_start
        
        return chunks if chunks else [ChunkInfo(
            doc_index=doc_index,
            chunk_index=0,
            title=title,
            content="[Empty Document]",
            url=url,
            start_char=0,
            end_char=0
        )]
    
    def _split_into_chunks_by_chars(self, text: str, doc_index: int, title: str, url: str) -> List[ChunkInfo]:
        """
        Split a document into chunks based on character count.
        Fallback method when tiktoken is not available.
        
        Args:
            text: The document text to split
            doc_index: Index of the source document
            title: Document title
            url: Document URL
            
        Returns:
            List of ChunkInfo objects
        """
        if not text or not text.strip():
            return [ChunkInfo(
                doc_index=doc_index,
                chunk_index=0,
                title=title,
                content="[Empty Document]",
                url=url,
                start_char=0,
                end_char=0
            )]
        
        chunks = []
        text_len = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to find a good break point
            if end < text_len:
                break_point = text.rfind('\n\n', start, end)
                if break_point == -1 or break_point <= start:
                    for sep in ['。', '！', '？', '. ', '! ', '? ', '\n']:
                        break_point = text.rfind(sep, start, end)
                        if break_point > start:
                            end = break_point + len(sep)
                            break
                else:
                    end = break_point + 2
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append(ChunkInfo(
                    doc_index=doc_index,
                    chunk_index=chunk_index,
                    title=title,
                    content=chunk_content,
                    url=url,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start >= text_len:
                break
            if start <= chunks[-1].start_char if chunks else 0:
                start = end
        
        return chunks if chunks else [ChunkInfo(
            doc_index=doc_index,
            chunk_index=0,
            title=title,
            content="[Empty Document]",
            url=url,
            start_char=0,
            end_char=0
        )]
    
    def _split_into_chunks(self, text: str, doc_index: int, title: str, url: str) -> List[ChunkInfo]:
        """
        Split a document into chunks using the configured method.
        
        Args:
            text: The document text to split
            doc_index: Index of the source document
            title: Document title
            url: Document URL
            
        Returns:
            List of ChunkInfo objects
        """
        if self.use_token_chunking and self.tokenizer:
            return self._split_into_chunks_by_tokens(text, doc_index, title, url)
        else:
            return self._split_into_chunks_by_chars(text, doc_index, title, url)
    
    def _load_from_cache(self) -> bool:
        """Load chunks and embeddings from cache."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT doc_index, chunk_index, title, content, url, start_char, end_char, embedding 
            FROM chunks ORDER BY doc_index, chunk_index
        """)
        rows = cursor.fetchall()
        
        if not rows:
            return False
        
        self.chunks = []
        embeddings_list = []
        
        for row in rows:
            doc_index, chunk_index, title, content, url, start_char, end_char, embedding_blob = row
            self.chunks.append(ChunkInfo(
                doc_index=doc_index,
                chunk_index=chunk_index,
                title=title,
                content=content,
                url=url,
                start_char=start_char,
                end_char=end_char
            ))
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings_list.append(embedding)
        
        self.embeddings = np.array(embeddings_list)
        logger.info(f"Loaded {len(self.chunks)} chunks from cache")
        return True
    
    def _save_to_cache(self):
        """Save chunks and embeddings to cache."""
        cursor = self.conn.cursor()
        
        # Clear existing chunks
        cursor.execute("DELETE FROM chunks")
        
        # Insert chunks with embeddings
        for chunk, embedding in zip(self.chunks, self.embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                """INSERT INTO chunks 
                   (doc_index, chunk_index, title, content, url, start_char, end_char, embedding) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk.doc_index, chunk.chunk_index, chunk.title, chunk.content, 
                 chunk.url, chunk.start_char, chunk.end_char, embedding_blob)
            )
        
        self.conn.commit()
        logger.info(f"Saved {len(self.chunks)} chunks to cache")
    
    def _is_qwen_embedding_model(self) -> bool:
        """Check if the current embedding model is a Qwen/Alibaba model."""
        model_lower = self.embedding_model.lower()
        return "text-embedding-v" in model_lower or "qwen" in model_lower
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from stored embeddings or use default."""
        if self.embeddings is not None and len(self.embeddings.shape) > 1:
            return self.embeddings.shape[1]
        # Default dimensions: Qwen text-embedding-v4 uses 1024, OpenAI uses 1536
        return 1024 if self._is_qwen_embedding_model() else 1536
    
    def _get_embedding(self, text: str, max_retries: int = 5) -> np.ndarray:
        """Get embedding for a single text with retry logic for rate limiting."""
        # Ensure text is a non-empty string with actual content
        if not text or not isinstance(text, str) or not text.strip():
            text = "[Empty]"
        
        # Truncate if still too long (shouldn't happen with proper chunking)
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        for attempt in range(max_retries):
            try:
                # Build request parameters
                request_params = {
                    "model": self.embedding_model,
                    "input": text
                }
                # For Qwen/Alibaba embedding API, explicitly set encoding_format to float
                # This is required by Alibaba's text-embedding-v4 API
                if self._is_qwen_embedding_model():
                    request_params["encoding_format"] = "float"
                
                response = self.client.embeddings.create(**request_params)
                return np.array(response.data[0].embedding)
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limiting or instance config errors
                if any(keyword in error_str for keyword in ['rate', 'limit', '429', 'pre-004', '实例配置', '限流']):
                    wait_time = min(2 ** attempt * 5, 120)  # Exponential backoff: 5, 10, 20, 40, 80 seconds, max 120
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to get embedding for query: {e}")
                    # Return zero vector on failure - use correct dimension
                    return np.zeros(self._get_embedding_dimension())
        
        logger.error(f"Failed to get embedding after {max_retries} retries")
        return np.zeros(self._get_embedding_dimension())
    
    def _get_embeddings_for_chunks(self, chunks: List[ChunkInfo], max_retries: int = 5, request_delay: float = 0.1) -> np.ndarray:
        """Get embeddings for all chunks one by one with rate limiting protection.
        
        Args:
            chunks: List of chunks to get embeddings for
            max_retries: Maximum number of retries for rate-limited requests
            request_delay: Delay between requests in seconds (default: 0.1s = 100ms)
        """
        all_embeddings = []
        total = len(chunks)
        consecutive_errors = 0
        
        # Determine embedding dimension based on model
        # Qwen text-embedding-v4 uses 1024, OpenAI uses 1536
        default_dim = 1024 if self._is_qwen_embedding_model() else 1536
        
        for i, chunk in enumerate(chunks):
            text = chunk.content
            
            # Ensure text is valid
            if not text or not isinstance(text, str) or not text.strip():
                text = "[Empty]"
            
            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"Generating embeddings: {i + 1}/{total}")
            
            success = False
            for attempt in range(max_retries):
                try:
                    # Build request parameters
                    request_params = {
                        "model": self.embedding_model,
                        "input": text
                    }
                    # For Qwen/Alibaba embedding API, explicitly set encoding_format to float
                    if self._is_qwen_embedding_model():
                        request_params["encoding_format"] = "float"
                    
                    response = self.client.embeddings.create(**request_params)
                    embedding = np.array(response.data[0].embedding)
                    all_embeddings.append(embedding)
                    success = True
                    consecutive_errors = 0
                    
                    # Update default_dim based on actual embedding dimension
                    if default_dim != len(embedding):
                        default_dim = len(embedding)
                    
                    # Add delay between requests to avoid rate limiting
                    if request_delay > 0:
                        time.sleep(request_delay)
                    break
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check for rate limiting or instance config errors
                    if any(keyword in error_str for keyword in ['rate', 'limit', '429', 'pre-004', '实例配置', '限流']):
                        consecutive_errors += 1
                        # Exponential backoff with longer waits for consecutive errors
                        base_wait = min(2 ** attempt * 5, 120)
                        extra_wait = min(consecutive_errors * 10, 60)
                        wait_time = base_wait + extra_wait
                        logger.warning(f"Rate limited on chunk {i}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    
                    # Check for token length errors
                    elif "maximum context length" in error_str or "token" in error_str:
                        logger.warning(f"Chunk {i} too long, truncating...")
                        text = text[:3000]
                        continue
                    
                    else:
                        logger.error(f"Failed to get embedding for chunk {i}: {e}")
                        break
            
            if not success:
                logger.error(f"Failed to get embedding for chunk {i} after {max_retries} retries")
                # Use the correct dimension for zero vector
                all_embeddings.append(np.zeros(default_dim))
        
        return np.array(all_embeddings)
    
    def load_from_db(self, db_path: str, readonly: bool = True) -> int:
        """
        Load documents directly from a pre-built SQLite database.
        This skips the JSON file entirely and loads embeddings from the db.
        
        By default, opens in read-only mode to allow multiple processes
        to read the same db file concurrently without conflicts.
        
        Args:
            db_path: Path to the .chunks.db file
            readonly: If True (default), open in read-only mode.
                     This allows multiple processes to read the same file.
            
        Returns:
            Number of chunks loaded
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        self.db_path = db_path
        self.current_file = db_path  # Use db path as current file
        
        # Initialize database connection in read-only mode by default
        # This allows multiple processes to read the same db file concurrently
        self._init_db(db_path, readonly=readonly)
        
        # Load directly from cache without validation
        logger.info(f"Loading directly from database (readonly={readonly}): {db_path}")
        if self._load_from_cache():
            logger.info(f"Successfully loaded {len(self.chunks)} chunks from database")
            return len(self.chunks)
        else:
            raise ValueError(f"Failed to load chunks from database: {db_path}")
    
    def load_documents(self, file_path: str) -> int:
        """
        Load documents from a long_context.json file and split into chunks.
        Uses SQLite cache if available and valid.
        
        If a .chunks.db file exists alongside the JSON file, it will be used
        as a cache to avoid regenerating embeddings.
        
        Args:
            file_path: Path to the long_context.json file or .chunks.db file
            
        Returns:
            Number of chunks created
        """
        # Check if this is actually a .db file path - load directly from db
        if file_path.endswith('.db') or file_path.endswith('.chunks.db'):
            return self.load_from_db(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.current_file = file_path
        self.db_path = self._get_db_path(file_path)
        
        # Check if db file exists and try to load from it directly in READ-ONLY mode
        # This allows using pre-built db files without needing to validate hash
        # and allows multiple processes to read the same db file concurrently
        if os.path.exists(self.db_path):
            try:
                # Open in read-only mode to allow concurrent access
                self._init_db(self.db_path, readonly=True)
                # Try to load from cache - if it has data, use it
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"Found existing database with {count} chunks, loading from cache (READ-ONLY): {self.db_path}")
                    if self._load_from_cache():
                        return len(self.chunks)
            except Exception as e:
                logger.warning(f"Failed to load from existing db in read-only mode, will try read-write: {e}")
                # Close the failed connection
                if self.conn:
                    self.conn.close()
                    self.conn = None
        
        # If we get here, we need to generate embeddings from JSON
        # Make sure file_path is a JSON file, not a db file
        if file_path.endswith('.db'):
            raise ValueError(f"Cannot generate embeddings from db file. DB file must already exist and be valid: {file_path}")
        
        file_hash = self._get_file_hash(file_path)
        
        # Initialize database
        self._init_db(self.db_path)
        
        # Check if cache is valid (for backward compatibility)
        if self._check_cache_valid(file_hash):
            logger.info(f"Loading from cache: {self.db_path}")
            if self._load_from_cache():
                return len(self.chunks)
        
        # Load documents from JSON
        logger.info(f"Loading documents from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.documents = data
        elif isinstance(data, dict) and 'documents' in data:
            self.documents = data['documents']
        else:
            raise ValueError("Invalid long_context.json format")
        
        logger.info(f"Loaded {len(self.documents)} documents, splitting into chunks...")
        
        # Split documents into chunks
        self.chunks = []
        for doc_index, doc in enumerate(self.documents):
            title = doc.get('title', '')
            url = doc.get('url', '')
            # Support both 'content' and 'page_body' field names
            content = doc.get('content', '') or doc.get('page_body', '')
            
            # Prepend title to content for better context
            full_text = f"{title}\n\n{content}" if title else content
            
            doc_chunks = self._split_into_chunks(full_text, doc_index, title, url)
            self.chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        logger.info(f"Generating embeddings for chunks...")
        
        # Generate embeddings for all chunks
        self.embeddings = self._get_embeddings_for_chunks(self.chunks)
        
        # Save to cache
        self._save_cache_metadata(file_hash)
        self._save_to_cache()
        
        logger.info(f"Embeddings generated and cached for {len(self.chunks)} chunks")
        return len(self.chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        diverse: bool = False,
        max_per_doc: int = 2
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            diverse: If True, ensure results come from different documents
            max_per_doc: Maximum chunks per document when diverse=True
            
        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Validate embedding dimensions match
        db_dim = self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else len(self.embeddings[0])
        query_dim = len(query_embedding)
        if db_dim != query_dim:
            raise ValueError(
                f"Embedding dimension mismatch: database has {db_dim}-dim embeddings, "
                f"but query embedding has {query_dim} dimensions. "
                f"This usually means the embedding model used for query ({self.embedding_model}) "
                f"is different from the model used to build the database. "
                f"For Qwen embeddings (1024-dim), ensure QWEN_EMBEDDING_* environment variables are set correctly. "
                f"For OpenAI embeddings (1536-dim), use the default settings."
            )

        # Calculate cosine similarity with zero-vector handling
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Avoid division by zero: set similarity to -1 for zero-norm embeddings
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = np.dot(self.embeddings, query_embedding) / (embedding_norms * query_norm)
            # Replace nan/inf with -1 (lowest possible similarity)
            similarities = np.where(np.isfinite(similarities), similarities, -1.0)

        # Get all indices sorted by similarity (excluding invalid ones)
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        doc_counts = {}  # Track how many chunks per document
        
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
                
            score = float(similarities[idx])
            if score < min_score:
                continue
            
            chunk = self.chunks[idx]
            
            # Apply diversity constraint
            if diverse:
                doc_idx = chunk.doc_index
                if doc_idx not in doc_counts:
                    doc_counts[doc_idx] = 0
                if doc_counts[doc_idx] >= max_per_doc:
                    continue
                doc_counts[doc_idx] += 1
            
            results.append(SearchResult(
                title=chunk.title,
                content=chunk.content,
                score=score,
                doc_index=chunk.doc_index,
                chunk_index=chunk.chunk_index,
                metadata={
                    'url': chunk.url,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
            ))
        
        return results
    
    def diverse_search(
        self,
        query: str,
        top_k: int = 10,
        min_docs: int = 5,
        max_per_doc: int = 2,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search with document diversity - ensures results come from multiple documents.
        
        This method prioritizes getting results from different documents first,
        then fills remaining slots with additional chunks from the same documents.
        
        Args:
            query: Search query
            top_k: Total number of results to return
            min_docs: Minimum number of different documents to include
            max_per_doc: Maximum chunks per document
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects from diverse documents
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Validate embedding dimensions match
        db_dim = self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else len(self.embeddings[0])
        query_dim = len(query_embedding)
        if db_dim != query_dim:
            raise ValueError(
                f"Embedding dimension mismatch: database has {db_dim}-dim embeddings, "
                f"but query embedding has {query_dim} dimensions. "
                f"This usually means the embedding model used for query ({self.embedding_model}) "
                f"is different from the model used to build the database. "
                f"For Qwen embeddings (1024-dim), ensure QWEN_EMBEDDING_* environment variables are set correctly. "
                f"For OpenAI embeddings (1536-dim), use the default settings."
            )
        
        # Calculate cosine similarity with zero-vector handling
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Avoid division by zero: set similarity to -1 for zero-norm embeddings
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = np.dot(self.embeddings, query_embedding) / (embedding_norms * query_norm)
            # Replace nan/inf with -1 (lowest possible similarity)
            similarities = np.where(np.isfinite(similarities), similarities, -1.0)
        
        # Get all indices sorted by similarity (excluding invalid ones)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Phase 1: Get best chunk from each document (up to min_docs)
        results = []
        seen_docs = set()
        doc_best_chunks = {}  # doc_index -> list of (idx, score)
        
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue
            
            chunk = self.chunks[idx]
            doc_idx = chunk.doc_index
            
            if doc_idx not in doc_best_chunks:
                doc_best_chunks[doc_idx] = []
            doc_best_chunks[doc_idx].append((idx, score))
        
        # Sort documents by their best chunk score
        doc_scores = [(doc_idx, chunks[0][1]) for doc_idx, chunks in doc_best_chunks.items()]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Phase 1: Add best chunk from top documents
        for doc_idx, _ in doc_scores:
            if len(results) >= top_k:
                break
            if len(seen_docs) >= min_docs and len(results) >= min_docs:
                break
            
            idx, score = doc_best_chunks[doc_idx][0]
            chunk = self.chunks[idx]
            
            results.append(SearchResult(
                title=chunk.title,
                content=chunk.content,
                score=score,
                doc_index=chunk.doc_index,
                chunk_index=chunk.chunk_index,
                metadata={
                    'url': chunk.url,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
            ))
            seen_docs.add(doc_idx)
        
        # Phase 2: Fill remaining slots with additional chunks (respecting max_per_doc)
        doc_counts = {doc_idx: 1 for doc_idx in seen_docs}
        
        for doc_idx, _ in doc_scores:
            if len(results) >= top_k:
                break
            
            for idx, score in doc_best_chunks[doc_idx][1:]:  # Skip first (already added)
                if len(results) >= top_k:
                    break
                if doc_counts.get(doc_idx, 0) >= max_per_doc:
                    break
                
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    title=chunk.title,
                    content=chunk.content,
                    score=score,
                    doc_index=chunk.doc_index,
                    chunk_index=chunk.chunk_index,
                    metadata={
                        'url': chunk.url,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    }
                ))
                doc_counts[doc_idx] = doc_counts.get(doc_idx, 0) + 1
        
        logger.info(f"Diverse search returned {len(results)} results from {len(seen_docs)} documents")
        return results
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        top_k: int = 10
    ) -> str:
        """
        Get concatenated context from top relevant chunks.
        
        Args:
            query: Search query
            max_tokens: Maximum approximate tokens in context
            top_k: Number of chunks to consider
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for result in results:
            # Include title only for first chunk of each document
            if result.chunk_index == 0:
                chunk_text = f"## {result.title}\n\n{result.content}\n\n"
            else:
                chunk_text = f"{result.content}\n\n"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents and chunks."""
        if not self.chunks:
            return {"status": "no_documents_loaded"}
        
        # Count unique documents
        unique_docs = set(chunk.doc_index for chunk in self.chunks)
        total_chars = sum(len(chunk.content) for chunk in self.chunks)
        
        return {
            "total_documents": len(unique_docs),
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "average_chunk_length": total_chars // len(self.chunks) if self.chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "cache_path": self.db_path,
            "embedding_model": self.embedding_model,
            "sample_titles": list(set(chunk.title[:50] for chunk in self.chunks[:10]))
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
