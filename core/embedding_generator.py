from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding generator with a specific model.
        
        Args:
            model_name: The name of the HuggingFace model to use for embeddings.
                      Default is 'all-MiniLM-L6-v2' which creates 384-dimensional vectors.
        """
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder="model_cache/embeddings"
        )
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text chunks to generate embeddings for.
                  Each chunk will get its own embedding vector.
        
        Returns:
            List of embedding vectors, one for each input text chunk.
            Each vector is a 384-dimensional array (for all-MiniLM-L6-v2 model).
        """
        print(f"\nGenerating embeddings for {len(texts)} chunks...")
        print(f"Each embedding will be a {self.embeddings_model.client.get_sentence_embedding_dimension()}-dimensional vector")
        
        # Generate embeddings for each chunk
        embeddings = self.embeddings_model.embed_documents(texts)
        
        # Print some statistics
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding shape: {len(embeddings[0])}")
        print(f"First embedding preview: {embeddings[0][:5]}...")
        
        return embeddings