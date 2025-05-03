# RAG System

A Retrieval-Augmented Generation (RAG) system built with Streamlit, featuring document processing, semantic search, and quality evaluation capabilities.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_chat
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
follow the uv_guide.md in docs folder
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the interface:
   - Upload documents in the "File Upload" tab
   - Perform searches in the "Search" tab
   - View traces in the "Trace Viewer" tab

## Project Structure

```
rag_chat/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── core/
│   ├── chunker.py        # Document chunking logic
│   ├── document_processor.py  # Document processing
│   ├── embedding_generator.py # Embedding generation
│   ├── faiss_store.py    # Vector storage
│   ├── llm_tracer.py     # LLM interaction tracing
│   └── sqlite_store.py   # SQLite database operations
├── components/
│   └── trace_viewer.py   # Trace visualization component
└── uploads/             # Directory for uploaded files
```

