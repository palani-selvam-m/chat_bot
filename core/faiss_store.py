import os
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from .sqlite_store import SQLiteStore
from transformers import MarianMTModel, MarianTokenizer
import time
from pathlib import Path
from tqdm import tqdm
import sys

class FAISSVectorStore:
    def __init__(self, index_path="faiss_index", model_name="all-MiniLM-L6-v2", 
                 dimension=384,  # Default for all-MiniLM-L6-v2
                 nlist=100,      # Number of clusters for PQ
                 nprobe=10,      # Number of clusters to visit during search
                 M=8,           # Number of connections per layer in HNSW
                 efConstruction=40,  # Size of the dynamic candidate list during construction
                 efSearch=16):       # Size of the dynamic candidate list during search
        self.index_path = index_path
        self.model_name = model_name
        
        # Set up model cache directory
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        print("\nInitializing models...")
        
        # Initialize embeddings model with cache
        print("1. Loading embeddings model...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(self.cache_dir / "embeddings")
        )
        
        self.sqlite_store = SQLiteStore()
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
        # Initialize translation models with cache and proper device handling
        print("2. Loading translation models...")
        try:
            # Initialize English translation model
            print("Loading English translation model...")
            self.translator_en = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-mul-en',
                cache_dir=str(self.cache_dir / "translator_en"),
                device_map="auto"  # Let the model decide the best device
            )
            self.translator_tokenizer_en = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-mul-en',
                cache_dir=str(self.cache_dir / "translator_en")
            )
            
            # Initialize multilingual translation model
            print("Loading multilingual translation model...")
            self.translator_mul = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-en-mul',
                cache_dir=str(self.cache_dir / "translator_mul"),
                device_map="auto"  # Let the model decide the best device
            )
            self.translator_tokenizer_mul = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-en-mul',
                cache_dir=str(self.cache_dir / "translator_mul")
            )
            print("✓ Translation models loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading translation models: {e}")
            print("Translation features will be disabled")
            self.translator_en = None
            self.translator_tokenizer_en = None
            self.translator_mul = None
            self.translator_tokenizer_mul = None

        # Load or initialize FAISS index
        print("\n3. Initializing FAISS index...")
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path, 
                    self.embeddings_model,
                    allow_dangerous_deserialization=True
                )
                print("✓ Loaded existing FAISS index")
                # Reindex to ensure synchronization
                self.reindex_all_documents()
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.vectorstore = None
                self._init_index()
                print("Created new FAISS index")
        else:
            self.vectorstore = None
            self._init_index()
            print("Created new FAISS index")

    def _init_index(self):
        """Initialize FAISS index"""
        # Create a simple L2 index with the known dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Create a dummy document to initialize the vectorstore
        dummy_doc = Document(
            page_content="dummy",
            metadata={"vector_id": -1}
        )
        
        # Initialize the vectorstore with the dummy document
        self.vectorstore = FAISS.from_documents(
            [dummy_doc],
            self.embeddings_model
        )
        
        # Clear the index and docstore
        self.vectorstore.index.reset()
        self.vectorstore.docstore._dict.clear()
        self.vectorstore.index_to_docstore_id.clear()
        
        # Initialize index to vector_id mapping
        self.index_to_vector_id = {}
        
        # Add dummy document to mapping with -1 as vector_id
        self.index_to_vector_id[0] = -1

    def add_documents(self, chunks_with_embeddings):
        """
        chunks_with_embeddings: List of dicts, each containing:
        {
            "text": ...,
            "chapter": ...,
            "embedding": ...,
            "source": ...
        }
        """
        print("\nProcessing documents for storage:")
        print(f"Number of chunks to process: {len(chunks_with_embeddings)}")
        
        # Get the next available vector_id from SQLite
        with self.sqlite_store.get_connection() as conn:
            cursor = conn.execute("SELECT MAX(vector_id) FROM chunks")
            max_id = cursor.fetchone()[0]
            next_vector_id = (max_id + 1) if max_id is not None else 0
        
        print(f"Starting with vector_id: {next_vector_id}")
        
        documents = []
        embeddings = []
        vector_ids = []

        print("\n1. Preparing documents and embeddings...")
        
        for i, entry in enumerate(tqdm(chunks_with_embeddings, desc="Processing chunks", file=sys.stdout)):
            try:
                # Use the next available vector_id
                vector_id = next_vector_id + i
                vector_ids.append(vector_id)

                # Debug: Print chunk info
                print(f"\nProcessing chunk {i+1}/{len(chunks_with_embeddings)}:")
                print(f"Vector ID: {vector_id}")
                print(f"Text length: {len(entry['text'])}")
                print(f"Embedding shape: {len(entry['embedding'])}")
                print(f"Chapter: {entry['chapter']}")
                print(f"Source: {entry['source']}")

                # Create document with metadata
                doc = Document(
                    page_content=entry["text"],
                    metadata={
                        "vector_id": vector_id,  # Use the same vector_id
                        "chapter": entry["chapter"],
                        "source": entry["source"],
                        "chunk_index": i
                    }
                )
                documents.append(doc)
                embeddings.append(entry["embedding"])

                # Prepare chunk data for SQLite
                chunk_data = {
                    "chunk_text": entry["text"],
                    "source_file": entry["source"],
                    "chapter": entry["chapter"],
                    "metadata": {
                        "filename": entry["source"],
                        "chunk_number": vector_id,  # Use the same vector_id
                        "chunk_size": len(entry["text"]),
                        "chunk_index": i,
                        "embedding_shape": len(entry["embedding"])
                    }
                }

                # Store in SQLite
                print(f"\nStoring chunk {vector_id} in SQLite...")
                try:
                    self.sqlite_store.store_chunk(vector_id, chunk_data)
                    # Verify storage
                    stored_chunk = self.sqlite_store.verify_chunk_storage(vector_id)
                    if stored_chunk:
                        print(f"✓ Successfully verified chunk {vector_id} in SQLite")
                    else:
                        raise Exception(f"Failed to verify chunk {vector_id} in SQLite")
                except Exception as e:
                    print(f"Error storing chunk {vector_id} in SQLite: {e}")
                    raise

            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                raise

        print("\n2. Converting embeddings to numpy array...")
        try:
            embeddings_array = np.array(embeddings).astype('float32')
            print(f"Embeddings array shape: {embeddings_array.shape}")
        except Exception as e:
            print(f"Error converting embeddings to numpy array: {e}")
            raise
        
        print("\n3. Adding to vector store...")
        try:
            if not self.vectorstore:
                print("Creating new FAISS index...")
                self._init_index()
            
            print("Adding documents to FAISS...")
            self.vectorstore.add_documents(documents)
            self.vectorstore.index.add(embeddings_array)
            
            # Update index to vector_id mapping
            current_size = self.vectorstore.index.ntotal - len(vector_ids)
            for i, vector_id in enumerate(vector_ids):
                self.index_to_vector_id[current_size + i] = vector_id
            
            print(f"Updated index size: {self.vectorstore.index.ntotal}")
        except Exception as e:
            print(f"Error adding to vector store: {e}")
            raise

        print("\n4. Saving index...")
        try:
            self.vectorstore.save_local(self.index_path)
            # Save the index to vector_id mapping
            mapping_path = os.path.join(self.index_path, "index_mapping.npy")
            np.save(mapping_path, self.index_to_vector_id)
            print("✓ Index and mapping saved successfully")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise
        
        print(f"\n✓ Successfully processed {len(documents)} chunks")
        
        # Verify all chunks were stored
        print("\nVerifying chunk storage...")
        for vector_id in vector_ids:
            stored_chunk = self.sqlite_store.verify_chunk_storage(vector_id)
            if not stored_chunk:
                print(f"Warning: Chunk {vector_id} not found in SQLite")
        
        return vector_ids

    def translate_to_english(self, text):
        """Translate text to English"""
        if not self.translator_en or not self.translator_tokenizer_en:
            print("Translation models not available")
            return text
            
        try:
            inputs = self.translator_tokenizer_en(text, return_tensors="pt", padding=True)
            translated = self.translator_en.generate(**inputs)
            return self.translator_tokenizer_en.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation to English: {e}")
            return text

    def translate_from_english(self, text, target_lang):
        """Translate text from English to target language"""
        if not self.translator_mul or not self.translator_tokenizer_mul:
            print("Translation models not available")
            return text
            
        try:
            inputs = self.translator_tokenizer_mul(text, return_tensors="pt", padding=True)
            translated = self.translator_mul.generate(**inputs)
            return self.translator_tokenizer_mul.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation from English: {e}")
            return text

    def similarity_search(self, query, k=5, target_lang=None):
        start_time = time.time()
        
        # Check if vectorstore is initialized
        if not self.vectorstore:
            print("No vectorstore initialized - no documents indexed yet")
            return {
                "chunks": [],
                "search_time": 0.0,
                "original_query": query,
                "translated_query": query if target_lang else None,
                "message": "No documents have been indexed yet"
            }
        
        print(f"\nPerforming similarity search for query: {query}")
        print(f"Target language: {target_lang if target_lang else 'English'}")
        
        # Translate query to English if needed
        original_query = query
        if target_lang:
            query = self.translate_to_english(query)
            print(f"Translated query: {query}")
        
        try:
            # Get query embedding
            print("\nGenerating query embedding...")
            query_embedding = self.embeddings_model.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search
            print("\nPerforming FAISS search...")
            distances, indices = self.vectorstore.index.search(query_vector, k)
            print(f"Found {len(indices[0])} results")
            
            if len(indices[0]) == 0:
                print("No results found")
                return {
                    "chunks": [],
                    "search_time": time.time() - start_time,
                    "original_query": original_query,
                    "translated_query": query if target_lang else None,
                    "message": "No relevant documents found"
                }
            
            # Convert FAISS indices to vector_ids using the mapping
            print(f"indices[0]: {indices[0]}")
            print(f"index_to_vector_id keys: {list(self.index_to_vector_id.keys())}")
            
            # Filter out dummy document (vector_id -1) and get valid vector_ids
            vector_ids = []
            for idx in indices[0]:
                vector_id = self.index_to_vector_id.get(int(idx))
                if vector_id is not None and vector_id != -1:  # Skip dummy document
                    vector_ids.append(vector_id)
            
            print(f"Retrieved vector_ids: {vector_ids}")
            
            if not vector_ids:
                print("No valid results found after filtering dummy document")
                return {
                    "chunks": [],
                    "search_time": time.time() - start_time,
                    "original_query": original_query,
                    "translated_query": query if target_lang else None,
                    "message": "No relevant documents found"
                }
            
            # Get chunks from SQLite using the correct vector_ids
            chunks = self.sqlite_store.get_chunks_by_ids(vector_ids)
            print(f"Retrieved {len(chunks)} chunks from SQLite")
            
            if not chunks:
                print("No chunks found in SQLite")
                return {
                    "chunks": [],
                    "search_time": time.time() - start_time,
                    "original_query": original_query,
                    "translated_query": query if target_lang else None,
                    "message": "No relevant documents found"
                }
            
            # Add similarity scores and translate if needed
            for chunk, distance in zip(chunks, distances[0]):
                # Convert distance to similarity score (1 / (1 + distance))
                chunk["similarity_score"] = float(1 / (1 + distance))
                if target_lang:
                    chunk["translated_text"] = self.translate_from_english(chunk["chunk_text"], target_lang)
                # Ensure source field is present
                if "source_file" in chunk:
                    chunk["source"] = chunk["source_file"]
            
            # Calculate search time
            search_time = time.time() - start_time
            print(f"\nSearch completed in {search_time:.2f} seconds")
            
            # Log the query and results with timing
            query_id = self.sqlite_store.log_query(original_query, chunks, search_time)
            
            return {
                "chunks": chunks,
                "search_time": search_time,
                "original_query": original_query,
                "translated_query": query if target_lang else None,
                "query_id": query_id  # Add query_id to the response
            }
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return {
                "chunks": [],
                "search_time": time.time() - start_time,
                "original_query": original_query,
                "translated_query": query if target_lang else None,
                "message": f"Error during search: {str(e)}"
            }

    def visualize_embeddings(self, n_samples=100):
        """Visualize embeddings using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Please install required packages: pip install scikit-learn matplotlib seaborn")
            return

        if not self.vectorstore or self.vectorstore.index.ntotal == 0:
            print("No embeddings available to visualize")
            return

        print("\nVisualizing embeddings:")
        print("1. Getting embeddings from FAISS...")
        
        # Get all vectors from FAISS
        n_vectors = min(n_samples, self.vectorstore.index.ntotal)
        vectors = np.zeros((n_vectors, self.dimension), dtype=np.float32)
        self.vectorstore.index.reconstruct_n(0, n_vectors, vectors)
        
        print("2. Applying t-SNE dimensionality reduction...")
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
        
        print("3. Creating visualization...")
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1], alpha=0.6)
        plt.title(f"t-SNE Visualization of {n_vectors} Embeddings")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        # Save plot
        plot_path = "embeddings_visualization.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✓ Visualization saved to {plot_path}")
        
        # Print statistics
        print("\nEmbedding Statistics:")
        print(f"Total vectors in index: {self.vectorstore.index.ntotal}")
        print(f"Vector dimension: {self.dimension}")
        print(f"Sample size: {n_vectors}")
        
        return plot_path

    def get_embedding_stats(self):
        """Get statistics about stored embeddings"""
        if not self.vectorstore:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "index_type": "Not initialized"
            }
        
        return {
            "total_vectors": self.vectorstore.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.vectorstore.index).__name__,
            "hnsw_connections": self.M,
            "ef_search": self.efSearch,
            "nprobe": self.nprobe
        }

    def clear_index(self):
        """Clear the FAISS index and reinitialize it"""
        print("\nClearing FAISS index...")
        try:
            # Delete the index file if it exists
            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)
                print(f"✓ Deleted existing index at {self.index_path}")
            
            # Delete any .pkl files in the index directory
            index_dir = Path(self.index_path)
            if index_dir.exists():
                for pkl_file in index_dir.glob("*.pkl"):
                    pkl_file.unlink()
                    print(f"✓ Deleted {pkl_file}")
            
            # Clear the index to vector_id mapping
            self.index_to_vector_id = {}
            
            # Reinitialize the index
            self.vectorstore = None
            self._init_index()
            print("✓ Created new empty index")
            
            # Clear SQLite database
            self.sqlite_store.clear_database()
            print("✓ Cleared SQLite database")
            
            return True
        except Exception as e:
            print(f"Error clearing FAISS index: {e}")
            raise

    def reindex_all_documents(self):
        """Reindex all documents to ensure FAISS and SQLite indices are synchronized"""
        print("\nReindexing all documents...")
        
        # Get all chunks from SQLite
        chunks = self.sqlite_store.get_all_chunks()
        if not chunks:
            print("No documents found in SQLite")
            return
        
        print(f"Found {len(chunks)} documents to reindex")
        
        # Clear existing FAISS index
        self.clear_index()
        
        # Prepare documents and embeddings
        documents = []
        embeddings = []
        vector_ids = []
        
        for i, chunk in enumerate(chunks):
            # Use the original vector_id from SQLite
            vector_id = chunk.get("vector_id", i)
            vector_ids.append(vector_id)
            
            # Create document with metadata
            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    "vector_id": vector_id,  # Use original vector_id
                    "chapter": chunk["chapter"],
                    "source": chunk["source_file"],
                    "chunk_index": i
                }
            )
            documents.append(doc)
            
            # Get embedding for the chunk
            embedding = self.embeddings_model.embed_query(chunk["chunk_text"])
            embeddings.append(embedding)
            
            # Update chunk in SQLite with original vector_id
            chunk_data = {
                "chunk_text": chunk["chunk_text"],
                "source_file": chunk["source_file"],
                "chapter": chunk["chapter"],
                "metadata": {
                    "filename": chunk["source_file"],
                    "chunk_number": vector_id,  # Use original vector_id
                    "chunk_size": len(chunk["chunk_text"]),
                    "chunk_index": i,
                    "embedding_shape": len(embedding)
                }
            }
            self.sqlite_store.store_chunk(vector_id, chunk_data)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS
        if not self.vectorstore:
            self._init_index()
        
        self.vectorstore.add_documents(documents)
        self.vectorstore.index.add(embeddings_array)
        
        # Update index to vector_id mapping
        current_size = self.vectorstore.index.ntotal - len(vector_ids)
        for i, vector_id in enumerate(vector_ids):
            self.index_to_vector_id[current_size + i] = vector_id
        
        # Save the index and mapping
        self.vectorstore.save_local(self.index_path)
        mapping_path = os.path.join(self.index_path, "index_mapping.npy")
        np.save(mapping_path, self.index_to_vector_id)
        
        print(f"✓ Successfully reindexed {len(documents)} documents")
        print(f"Index to vector_id mapping: {self.index_to_vector_id}")
