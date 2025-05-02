from typing import Dict, Any, List
from .llm_tracer import LLMTracer
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self):
        self.tracer = LLMTracer()
        
        # Create the RAG prompt template
        self.rag_template = """You are a helpful AI assistant. Use the following retrieved documents to answer the question.
        If you cannot find the answer in the documents, say so.

        Retrieved documents:
        {context}

        Question: {question}

        Answer: Let me help you with that."""
        
        # Create the chain with tracing
        self.chain = self.tracer.create_chain(
            prompt_template=self.rag_template,
            output_key="answer"
        )
    
    def process_query(
        self,
        question: str,
        retrieved_documents: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a query using the RAG chain with tracing.
        
        Args:
            question: The user's question
            retrieved_documents: List of retrieved documents with their content
            metadata: Optional metadata for tracing (e.g., document IDs, retrieval scores)
        
        Returns:
            Dict containing the answer and any additional information
        """
        # Format the context from retrieved documents
        context = "\n\n".join(
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_documents)
        )
        
        # Add retrieval metadata if provided
        trace_metadata = {
            "query_type": "rag",
            "num_documents": len(retrieved_documents),
            **(metadata or {})
        }
        
        # Get response with tracing
        response = self.tracer.get_traced_response(
            chain=self.chain,
            inputs={
                "context": context,
                "question": question
            },
            metadata=trace_metadata
        )
        
        return response

# Example usage:
"""
rag = RAGChain()

# Process a query
response = rag.process_query(
    question="What are the key points about X?",
    retrieved_documents=[
        {"content": "Document content about X..."},
        {"content": "More information about X..."}
    ],
    metadata={
        "retrieval_method": "faiss",
        "similarity_scores": [0.95, 0.85]
    }
)

print(response["answer"])
""" 