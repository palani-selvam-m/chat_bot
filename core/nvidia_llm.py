from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

class NVIDIAChunkAnalyzer:
    def __init__(self, model_name="meta/llama3-8b-instruct"):
        self.llm = ChatNVIDIA(model=model_name)  # Uses NVIDIA API key from env var

        self.prompt_template = ChatPromptTemplate.from_template("""
            You are a legal language model. Perform the following on the given legal text:

            1. Rewrite it in plain English while keeping the legal structure (e.g., section numbers, clauses).
            2. Summarize the chunk in 2-3 sentences.
            3. Extract the last sentence or clause as the `end_context`.
            
            Respond only with a JSON object like below. 
            All values must be **strings** — do not return arrays or sets. 
            Do not add any explanation or introductory text.
            
            Format:
            {{
              "rewritten_chunk": "...",
              "summary": "...",
              "end_context": "..."
            }}
            
            Chunk:
            \"\"\"{chunk}\"\"\"
        """)

        self.output_parser = StrOutputParser()

    def analyze_chunk(self, chunk: str):
        chain = self.prompt_template | self.llm | self.output_parser
        response = chain.invoke({"chunk": chunk})
        try:
            import json
            return json.loads(response)
        except Exception:
            return {"rewritten_chunk": "", "summary": "", "end_context": "", "error": response}
