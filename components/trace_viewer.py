import streamlit as st
import json
from typing import Dict, Any

class TraceViewer:
    def __init__(self):
        self.sqlite_store = None

    def set_store(self, store):
        self.sqlite_store = store

    def render(self):
        """Render the trace viewer component"""
        st.title("Trace Viewer")
        
        # Get query history
        queries = self.sqlite_store.get_query_history()
        
        if not queries:
            st.info("No queries have been logged yet.")
            return
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Query History", "Detailed Traces"])
        
        with tab1:
            # Display recent queries
            st.subheader("Recent Queries")
            for query in queries:
                with st.expander(f"Query: {query['query_text']} ({query['created_at']})"):
                    # Display query details
                    st.write("Original Query:", query['query_text'])
                    if query['translated_query']:
                        st.write("Translated Query:", query['translated_query'])
                    st.write("Search Time:", f"{query['search_time']:.2f}s")
                    
                    # Display RAGAS metrics if available
                    if query['ragas_metrics']:
                        try:
                            metrics = json.loads(query['ragas_metrics']) if isinstance(query['ragas_metrics'], str) else query['ragas_metrics']
                            st.markdown("#### RAGAS Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.2f}")
                            with col2:
                                st.metric("Relevance", f"{metrics.get('relevance', 0):.2f}")
                            with col3:
                                st.metric("Context Precision", f"{metrics.get('context_precision', 0):.2f}")
                            with col4:
                                st.metric("Context Recall", f"{metrics.get('context_recall', 0):.2f}")
                            
                            if 'message' in metrics:
                                st.info(metrics['message'])
                        except Exception as e:
                            st.warning(f"Could not parse RAGAS metrics: {e}")
        
        with tab2:
            # Allow selection of a query to view its traces
            query_options = {f"{q['query_text']} ({q['created_at']})": q['id'] for q in queries}
            selected_query = st.selectbox("Select a query to view traces:", list(query_options.keys()))
            
            if selected_query:
                query_id = query_options[selected_query]
                traces = self.sqlite_store.get_llm_traces(query_id)
                
                if traces:
                    st.subheader("LLM Traces")
                    for trace in traces:
                        st.markdown("---")
                        st.markdown(f"**Trace {trace['id']}** ({trace['created_at']})")
                        self.render_trace_details(trace)
                else:
                    st.info("No traces available for this query.")

    def render_trace_details(self, trace: Dict[str, Any]):
        """Render detailed information for a single trace."""
        st.markdown("### Trace Details")
        
        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", trace.get("status", "unknown"))
        with col2:
            st.metric("Duration", f"{trace.get('execution_time', 0):.2f}s")
        
        # Display prompt and response
        st.markdown("#### Prompt")
        st.code(trace.get('prompt', ''), language="text")
        
        st.markdown("#### Response")
        st.write(trace.get('response', ''))
        
        # Display metadata if available
        if trace.get('metadata'):
            try:
                metadata = json.loads(trace['metadata']) if isinstance(trace['metadata'], str) else trace['metadata']
                st.markdown("#### Metadata")
                st.json(metadata)
            except Exception as e:
                st.warning(f"Could not parse metadata: {e}")
