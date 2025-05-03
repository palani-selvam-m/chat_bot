import streamlit as st
from components.trace_viewer import TraceViewer
from core.evaluator import RagasEvaluator
from core.llm_tracer import LLMTracer
import os
from pathlib import Path
from core.document_processor import DocumentProcessor
from core.chunker import ChapterAwareChunker
from core.embedding_generator import EmbeddingGenerator
from core.faiss_store import FAISSVectorStore
from core.sqlite_store import SQLiteStore
import time
import re
from tqdm import tqdm
import sys
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas.metrics import AspectCritic
from langchain_core.messages import AIMessage

# Create evaluation LLM
eval_llm = ChatNVIDIA(
    model="mixtral_8x7b",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY")
)

# Initialize metrics with AspectCritic
faithfulness = AspectCritic(
    name="faithfulness",
    llm=eval_llm,
    definition="Verify if the response is factually consistent with the context."
)

relevance = AspectCritic(
    name="relevance",
    llm=eval_llm,
    definition="Verify if the response directly answers the question."
)

context_precision = AspectCritic(
    name="context_precision",
    llm=eval_llm,
    definition="Verify if the retrieved context is relevant to the question."
)

context_recall = AspectCritic(
    name="context_recall",
    llm=eval_llm,
    definition="Verify if all relevant information from the context is used in the response."
)

def initialize_session_state():
    """Initialize session state variables."""
    if "llm_tracer" not in st.session_state:
        st.session_state.llm_tracer = LLMTracer()
    if "sqlite_store" not in st.session_state:
        st.session_state.sqlite_store = SQLiteStore()
    if "trace_viewer" not in st.session_state:
        st.session_state.trace_viewer = TraceViewer()
        st.session_state.trace_viewer.set_store(st.session_state.sqlite_store)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISSVectorStore()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "evaluator" not in st.session_state:
        st.session_state.rag_tracer = RagasEvaluator()

def process_uploaded_file(file):
    """Process an uploaded file and store it."""
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save the file
    file_path = upload_dir / file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    try:
        print(f"\nProcessing file: {file.name}")
        print("=" * 50)
        
        # Process the document
        print("Step 1/4: Extracting text from document...")
        processor = DocumentProcessor(file_path)
        full_text = processor.extract_text()
        print(f"✓ Text extraction complete. Extracted {len(full_text)} characters")
        print(f"First 100 characters: {full_text[:100]}...")
        
        # Chunk the text
        print("\nStep 2/4: Chunking document...")
        chunker = ChapterAwareChunker()
        raw_chunks = chunker.chunk(full_text)
        print(f"✓ Chunking complete. Created {len(raw_chunks)} chunks")
        print(f"First chunk preview: {raw_chunks[0][:100]}...")
        
        # Convert chunks to dictionaries
        print("\nStep 3/4: Processing chunks...")
        chapter_chunks = []
        for i, chunk in enumerate(tqdm(raw_chunks, desc="Converting chunks", file=sys.stdout)):
            # Extract chapter information if present
            chapter_match = re.search(r"CHAPTER\s+[IVXLCDM]+", chunk, re.IGNORECASE)
            chapter = chapter_match.group(0) if chapter_match else "Unknown"
            
            chapter_chunks.append({
                "text": chunk,
                "chapter": chapter
            })
            if i == 0:  # Print details of first chunk
                print(f"\nFirst chunk details:")
                print(f"Chapter: {chapter}")
                print(f"Length: {len(chunk)} characters")
                print(f"Preview: {chunk[:100]}...")
        
        # Generate embeddings and combine with chunks
        print("\nStep 4/4: Generating embeddings...")
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(
            [chunk["text"] for chunk in tqdm(chapter_chunks, desc="Generating embeddings", file=sys.stdout)]
        )
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding shape: {len(embeddings[0])}")
        
        # Combine chunks with their embeddings
        processed_chunks = []
        for chunk, embedding in zip(chapter_chunks, embeddings):
            chunk["embedding"] = embedding
            chunk["source"] = file.name
            processed_chunks.append(chunk)
        
        # Store in FAISS
        print("\nStoring in vector database...")
        vector_ids = st.session_state.vector_store.add_documents(processed_chunks)
        print(f"Stored {len(vector_ids)} chunks in vector database")
        
        # Add to processed files only after successful processing
        st.session_state.processed_files.add(file.name)
        
        print("\nProcessing Summary:")
        print(f"- Total chunks: {len(processed_chunks)}")
        print(f"- Unique chapters: {len(set(chunk['chapter'] for chunk in processed_chunks))}")
        print(f"- Average chunk length: {sum(len(chunk['text']) for chunk in processed_chunks) / len(processed_chunks):.0f} characters")
        print("=" * 50)
        
        return {
            "file_path": file_path,
            "chunks": processed_chunks
        }
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)}")
        # Remove the file if processing failed
        if file_path.exists():
            file_path.unlink()
        return None

async def evaluate_rag_quality(query, response, context):
    """Evaluate RAG quality using RAGAS metrics."""
    try:
        # Extract text content from response
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = str(response)

        # If no context is provided, return a message
        if not context or context.strip() == "":
            return {
                'faithfulness': 0.0,
                'relevance': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'message': 'No relevant context found for evaluation'
            }

        print("\nEvaluating RAG quality...")
        print(f"Query: {query}")
        print(f"Response: {response_text[:100]}...")
        print(f"Context length: {len(context)} characters")

        # Create test data for evaluation
        test_data = {
            "user_input": query,
            "response": response_text,
            "context": context
        }

        # Calculate metrics
        metrics = {}
        
        # Create evaluation chain
        eval_chain = st.session_state.llm_tracer.create_chain(
            prompt_template="""You are an expert evaluator. Evaluate the following response based on the given context and question.

Question: {user_input}
Context: {context}
Response: {response}

Evaluate the following aspects and provide a score between 0 and 1:
1. Faithfulness: How factually consistent is the response with the context?
2. Relevance: How well does the response answer the question?
3. Context Precision: How relevant is the retrieved context to the question?
4. Context Recall: How well does the response use all relevant information from the context?

Provide your evaluation in this format:
faithfulness: [score]
relevance: [score]
context_precision: [score]
context_recall: [score]"""
        )
        
        # Get evaluation
        evaluation = st.session_state.llm_tracer.invoke_chain(
            chain=eval_chain,
            inputs=test_data
        )
        
        # Extract scores from evaluation
        if isinstance(evaluation, AIMessage):
            eval_text = evaluation.content
        else:
            eval_text = str(evaluation)
            
        # Parse scores with more robust parsing
        for line in eval_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Try different formats
            if ':' in line:
                parts = line.split(':', 1)  # Split on first colon only
                if len(parts) == 2:
                    metric, score = parts
                    metric = metric.strip().lower()
                    try:
                        # Extract first number found in the score part
                        score_match = re.search(r'[-+]?\d*\.\d+|\d+', score)
                        if score_match:
                            score = float(score_match.group())
                            metrics[metric] = min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
                    except (ValueError, AttributeError):
                        continue
            elif '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    metric, score = parts
                    metric = metric.strip().lower()
                    try:
                        score_match = re.search(r'[-+]?\d*\.\d+|\d+', score)
                        if score_match:
                            score = float(score_match.group())
                            metrics[metric] = min(max(score, 0.0), 1.0)
                    except (ValueError, AttributeError):
                        continue
        
        # Ensure all metrics are present
        required_metrics = ['faithfulness', 'relevance', 'context_precision', 'context_recall']
        for metric in required_metrics:
            if metric not in metrics:
                metrics[metric] = 0.0
        
        print("\nFinal metrics:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.2f}")
        
        return metrics

    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        # Return default metrics with error message
        return {
            'faithfulness': 0.0,
            'relevance': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'message': f'Error evaluating metrics: {str(e)}'
        }

def display_search_results(search_result):
    """Display search results in a formatted way"""
    if not search_result["chunks"]:
        st.info("No relevant documents found.")
        return
    # Display search statistics
    st.caption(f"Search completed in {search_result['search_time']:.2f} seconds")
    if search_result.get("translated_query"):
        st.caption(f"Translated query: {search_result['translated_query']}")

    # Display each chunk
    for i, doc in enumerate(search_result["chunks"], 1):
        with st.expander(f"Result {i} (Score: {doc.get('similarity_score', 0):.2f})"):
            # Display source and chapter information
            source = doc.get("source", doc.get("source_file", "Unknown source"))
            chapter = doc.get("chapter", "Unknown chapter")
            st.caption(f"Source: {source} | Chapter: {chapter}")
            
            # Display the text content
            if "translated_text" in doc:
                st.write("Original text:")
                st.write(doc["chunk_text"])
                st.write("Translated text:")
                st.write(doc["translated_text"])
            else:
                st.write(doc["chunk_text"])

def clear_databases():
    """Clear both FAISS and SQLite databases"""
    try:
        # Clear SQLite database
        st.session_state.sqlite_store.clear_database()
        
        # Clear FAISS index
        st.session_state.vector_store.clear_index()
        
        # Clear processed files
        st.session_state.processed_files.clear()
        st.session_state.uploaded_files.clear()
        
        st.success("Successfully cleared all databases!")
    except Exception as e:
        st.error(f"Error clearing databases: {str(e)}")

async def main():
    st.set_page_config(
        page_title="RAG System",
        layout="wide"
    )
    
    st.title("RAG System")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["File Upload", "Search", "Trace Viewer"])
    
    with tab1:
        st.subheader("Upload Documents")
        
        # Add clear database button
        if st.button("Clear All Databases", type="secondary"):
            if st.checkbox("I understand this will delete all stored data"):
                clear_databases()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {file.name}..."):
                        result = process_uploaded_file(file)
                        if result:
                            st.success(f"Successfully processed {file.name}")
                            st.session_state.uploaded_files.append({
                                "name": file.name,
                                "path": str(result["file_path"]),
                                "chunks": result["chunks"]
                            })
                        else:
                            st.error(f"Failed to process {file.name}")
        
        # Display processed files
        if st.session_state.uploaded_files:
            st.subheader("Processed Files")
            for file_info in st.session_state.uploaded_files:
                st.write(f"✓ {file_info['name']}")
    
    with tab2:
        st.subheader("Search Documents")
        
        # Search input
        query = st.text_input("Enter your search query:")
        
        # Language selection
        target_lang = st.selectbox(
            "Select response language",
            options=["English", "Hindi", "Spanish", "French", "German"],
            index=0  # This sets English as the default
        )
        
        # Search options
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results", 1, 10, 3)
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)
        
        if query:
            with st.spinner("Searching..."):
                # Perform search using our custom FAISSVectorStore implementation
                search_result = st.session_state.vector_store.similarity_search(
                    query=query,
                    k=num_results,
                    target_lang=target_lang
                )
                
                # Display search results
                display_search_results(search_result)
                
                # Get chunks and prepare context
                chunks = search_result.get("chunks", [])
                
                # Debug information
                if st.checkbox("Show Debug Info"):
                    st.write("Search Result:", search_result)
                    st.write("Number of chunks found:", len(chunks))
                
                # Check if we found any relevant documents
                if not chunks:
                    st.warning("No relevant documents found for your query. Please try a different search term or upload relevant documents.")
                    return
                
                context = "\n\n".join(chunk["chunk_text"] for chunk in chunks)
                
                # Create a chain for the search
                prompt_template = """You are a helpful AI assistant. Your task is to answer questions based on the provided context.

                Context:
                {context}

                Question: {question}

                Instructions:
                1. Answer the question using ONLY the information from the context above
                2. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context"
                3. Keep your answer concise and relevant
                4. Include relevant quotes or references from the context when appropriate
                5. If the question is unclear, ask for clarification

                Answer:"""
                
                search_chain = st.session_state.llm_tracer.create_chain(
                    prompt_template=prompt_template
                )
                
                # Run the chain with tracing
                start_time = time.time()
                response = st.session_state.llm_tracer.invoke_chain(
                    chain=search_chain,
                    inputs={
                        "context": context,
                        "question": query
                    }
                )
                
                # Calculate search time
                search_time = time.time() - start_time
                
                # Get and store traces
                traces = st.session_state.llm_tracer.get_traces()
                for trace in traces:
                    st.session_state.sqlite_store.log_llm_trace(
                        query_id=search_result.get("query_id"),
                        prompt=trace.get("inputs", {}).get("prompt", ""),
                        response=str(trace.get("outputs", {}).get("response", "")),
                        execution_time=trace.get("duration", 0),
                        metadata={"trace_type": "llm_response"}
                    )
                
                # Extract response text
                if isinstance(response, str):
                    response_text = response
                elif isinstance(response, AIMessage):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Translate response if language is not English
                if target_lang != "English":
                    translation_prompt = f"""Translate the following text to {target_lang}. Maintain the original meaning and format:

                    {response_text}

                    Translation:"""
                    
                    translation_chain = st.session_state.llm_tracer.create_chain(
                        prompt_template=translation_prompt
                    )
                    
                    translated_response = st.session_state.llm_tracer.invoke_chain(
                        chain=translation_chain,
                        inputs={}
                    )
                    
                    if isinstance(translated_response, AIMessage):
                        response_text = translated_response.content
                    else:
                        response_text = str(translated_response)
                
                # Display response
                st.subheader("Response")
                st.write(response_text)
                
                # Evaluate RAG quality
                metrics = await evaluate_rag_quality(query, response_text, context)
                
                # Display metrics
                st.subheader("RAG Quality Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Faithfulness", f"{metrics['faithfulness']:.2f}")
                with col2:
                    st.metric("Relevance", f"{metrics['relevance']:.2f}")
                with col3:
                    st.metric("Context Precision", f"{metrics['context_precision']:.2f}")
                with col4:
                    st.metric("Context Recall", f"{metrics['context_recall']:.2f}")
                
                if 'message' in metrics:
                    st.info(metrics['message'])
    
    with tab3:
        st.session_state.trace_viewer.render()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 