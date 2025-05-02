from langfuse import Langfuse
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

class LangfuseTracer:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

    def trace_query(self, query: str, results: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        trace = self.langfuse.trace(
            name="query_trace",
            metadata=metadata or {}
        )

        # Log the query
        trace.span(
            name="query_processing",
            input={"query": query},
            output={"results": results}
        )

        # Log each retrieved chunk
        for i, result in enumerate(results):
            trace.span(
                name=f"chunk_{i}",
                input={"chunk_id": result.get("vector_id")},
                output={
                    "chunk_text": result.get("chunk_text"),
                    "source_file": result.get("source_file"),
                    "similarity_score": result.get("similarity_score", 0.0)
                }
            )

        trace.end()
        return trace.id

    def trace_evaluation(self, query: str, results: List[Dict[str, Any]], metrics: Dict[str, float]):
        trace = self.langfuse.trace(
            name="evaluation_trace",
            metadata={"query": query}
        )

        # Log evaluation metrics
        trace.span(
            name="ragas_evaluation",
            input={"query": query, "results": results},
            output={"metrics": metrics}
        )

        trace.end()
        return trace.id 