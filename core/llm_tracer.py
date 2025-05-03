from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.runnables import RunnableSequence

load_dotenv()

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.traces = []
        self.current_trace = None
        self.current_question = ""
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts."""
        if serialized is None:
            serialized = {"name": "unknown"}
        self.current_trace = {
            "timestamp": datetime.now().timestamp(),
            "type": "chain",
            "name": serialized.get("name", "unknown"),
            "inputs": inputs,
            "status": "started"
        }
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        if self.current_trace:
            self.current_trace.update({
                "outputs": outputs,
                "status": "completed",
                "duration": datetime.now().timestamp() - self.current_trace["timestamp"]
            })
            self.traces.append(self.current_trace)
            self.current_trace = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        self.current_trace = {
            "timestamp": datetime.now().timestamp(),
            "inputs": {
                "prompt": prompts[0] if prompts else "",
                "question": self.current_question
            },
            "status": "started"
        }
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Run when LLM ends."""
        if self.current_trace:
            self.current_trace.update({
                "outputs": {"response": response},
                "status": "completed",
                "duration": datetime.now().timestamp() - self.current_trace["timestamp"]
            })
            self.traces.append(self.current_trace)
            self.current_trace = None
            self.current_question = ""
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all traces."""
        return self.traces
    
    def clear_traces(self):
        """Clear all traces."""
        self.traces = []
        self.current_trace = None
        self.current_question = ""

class LLMTracer:
    def __init__(self):
        # Initialize custom callback handler
        self.callback_handler = StreamlitCallbackHandler()
        
        # Create callback manager
        self.callback_manager = CallbackManager([self.callback_handler])
        
        # Initialize NVIDIA AI endpoint
        self.llm = ChatNVIDIA(
            model="mixtral_8x7b",
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            callback_manager=self.callback_manager
        )
    
    def create_chain(self, prompt_template: str) -> RunnableSequence:
        """Create a chain with the given prompt template."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm
    
    def invoke_chain(self, chain: RunnableSequence, inputs: Dict[str, Any]) -> Any:
        """Invoke the chain with the given inputs."""
        # Store the question in the callback handler for tracing
        self.callback_handler.current_question = inputs.get("question", "")
        return chain.invoke(inputs)
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all traces."""
        return self.callback_handler.get_traces()
    
    def clear_traces(self):
        """Clear all traces."""
        self.callback_handler.clear_traces()

