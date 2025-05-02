from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    # context_relevancy,
    context_recall,
    context_precision
)
from ragas import evaluate
from datasets import Dataset
from typing import List, Dict, Any
# from .langfuse_tracer import LangfuseTracer

class RagasEvaluator:
    # def __init__(self):
    #     self.tracer = LangfuseTracer()

    def evaluate_results(self, query: str, results: List[Dict[str, Any]], ground_truth: str = None):
        # Prepare data for Ragas
        data = {
            "question": [query],
            "answer": [ground_truth or ""],  # If no ground truth, use empty string
            "contexts": [[result["chunk_text"] for result in results]],
            "ground_truths": [[ground_truth]] if ground_truth else [[]]
        }
        
        dataset = Dataset.from_dict(data)

        # Define metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            # context_relevancy,
            context_recall,
            context_precision
        ]

        # Run evaluation
        result = evaluate(dataset, metrics)

        # Convert to dictionary
        metrics_dict = {
            metric.name: score 
            for metric, score in zip(metrics, result)
        }

        # Log to Langfuse
        # self.tracer.trace_evaluation(query, results, metrics_dict)

        return metrics_dict 