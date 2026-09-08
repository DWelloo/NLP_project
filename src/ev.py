"""
Evaluation Service for RAG System using RAGAS metrics.

Provides Faithfulness, Answer Relevancy, and Context Precision metrics
for evaluating the quality of the hybrid RAG system.

Uses Gemini 2.5 Flash Lite for evaluation.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import pandas as pd
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.logger import get_logger
from src.rag_service import RAGService
from src.config import GEMINI_API_KEY, EMBEDDINGS_MODEL


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    question: str
    ground_truth: str
    generated_answer: str
    contexts: list[str]
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float


class EvaluationService:
    """Service for evaluating RAG system quality using RAGAS metrics."""

    # Use Gemini 2.5 Flash Lite for evaluation
    EVAL_MODEL_NAME = "gemini-2.0-flash-lite"

    # Configuration for API
    REQUEST_TIMEOUT = 120
    MAX_RETRIES = 3
    MAX_WORKERS = 1  # Single worker to minimize rate limiting

    def __init__(self, rag_service: RAGService):
        self._logger = get_logger("EvaluationService")
        self._rag = rag_service

        # Use Gemini Flash Lite for RAGAS
        self._logger.info(
            f"Initializing Gemini LLM for RAGAS (model={self.EVAL_MODEL_NAME})..."
        )
        langchain_llm = ChatGoogleGenerativeAI(
            model=self.EVAL_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            timeout=self.REQUEST_TIMEOUT,
            max_retries=self.MAX_RETRIES,
        )
        self._llm = LangchainLLMWrapper(langchain_llm)

        # Use LangChain embeddings for RAGAS
        self._logger.info("Initializing embeddings for RAGAS evaluation...")
        langchain_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self._embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        self._test_dataset_path = (
            Path(__file__).parent.parent / "tests" / "test_dataset.json"
        )

    def load_test_dataset(self) -> list[dict]:
        """Load test dataset from JSON file."""
        self._logger.info(f"Loading test dataset from {self._test_dataset_path}")
        with open(self._test_dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_evaluation(
        self,
        use_vector: bool = True,
        use_graph: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run RAGAS evaluation on the test dataset.

        Args:
            use_vector: Whether to use vector search
            use_graph: Whether to use graph search
            limit: Optional limit on number of questions to evaluate

        Returns:
            DataFrame with evaluation results
        """
        self._logger.info("=" * 50)
        self._logger.info(f"Starting RAGAS evaluation with {self.EVAL_MODEL_NAME}...")
        self._logger.info(
            f"Settings: use_vector={use_vector}, use_graph={use_graph}, limit={limit}"
        )
        self._logger.info("=" * 50)

        test_data = self.load_test_dataset()
        if limit:
            test_data = test_data[:limit]

        self._logger.info(f"Evaluating {len(test_data)} questions...")

        # Collect RAG responses for each question
        questions = []
        ground_truths = []
        answers = []
        contexts = []

        # Use a separate LLM instance for RAG queries to use flash-lite
        from langchain_google_genai import ChatGoogleGenerativeAI as EvalLLM

        eval_llm = EvalLLM(
            model=self.EVAL_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0
        )

        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            self._logger.info(
                f"[{i+1}/{len(test_data)}] Processing: {question[:50]}..."
            )

            try:
                # Get RAG response - but we can't change the RAG service's LLM easily
                # So we'll just use the existing RAG service
                results = self._rag.route_and_respond(
                    question, use_vector=use_vector, use_graph=use_graph
                )

                # Extract answer (last result is the hybrid answer)
                generated_answer = str(results[-1].response) if results else ""
                self._logger.debug(f"  -> Answer length: {len(generated_answer)} chars")

                # Extract contexts - improved parsing
                context_list = self._extract_contexts(results, use_vector, use_graph)

                self._logger.debug(f"  -> Contexts found: {len(context_list)}")

                questions.append(question)
                ground_truths.append(ground_truth)
                answers.append(generated_answer)
                contexts.append(
                    context_list if context_list else ["Brak dostępnego kontekstu"]
                )

            except Exception as e:
                self._logger.error(f"  -> Error processing question: {e}")
                questions.append(question)
                ground_truths.append(ground_truth)
                answers.append(f"Error: {str(e)}")
                contexts.append(["Błąd przetwarzania"])

        # Create RAGAS dataset
        self._logger.info("Creating RAGAS dataset...")
        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "ground_truth": ground_truths,
                "answer": answers,
                "contexts": contexts,
            }
        )

        self._logger.info("=" * 50)
        self._logger.info(f"Running RAGAS metrics with {self.EVAL_MODEL_NAME}...")
        self._logger.info(
            f"Total evaluations: {len(test_data) * 3} (3 metrics per question)"
        )
        self._logger.info("=" * 50)

        # Configure RAGAS run settings
        run_config = RunConfig(
            timeout=self.REQUEST_TIMEOUT,
            max_retries=self.MAX_RETRIES,
            max_workers=self.MAX_WORKERS,
            max_wait=60,
        )

        # Run evaluation with RAGAS
        try:
            results = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=self._llm,
                embeddings=self._embeddings,
                run_config=run_config,
                raise_exceptions=False,  # Don't crash on individual failures
            )

            # Convert to DataFrame
            df = results.to_pandas()
            df.insert(0, "question", questions)
            df.insert(1, "ground_truth", ground_truths)
            df.insert(2, "answer", answers)

            self._logger.info("=" * 50)
            self._logger.info("Evaluation complete!")
            self._logger.info(f"Results shape: {df.shape}")
            self._logger.info("=" * 50)

            return df

        except Exception as e:
            self._logger.error(f"RAGAS evaluation failed: {e}")
            import traceback

            self._logger.error(traceback.format_exc())

            # Return DataFrame with just the collected data (without metrics)
            return pd.DataFrame(
                {
                    "question": questions,
                    "ground_truth": ground_truths,
                    "answer": answers,
                    "faithfulness": [None] * len(questions),
                    "answer_relevancy": [None] * len(questions),
                    "context_precision": [None] * len(questions),
                }
            )

    def _extract_contexts(
        self, results, use_vector: bool, use_graph: bool
    ) -> list[str]:
        """Extract and clean context from RAG results."""
        context_list = []

        if use_vector and len(results) > 0:
            vector_context = results[0].response
            if vector_context and "Brak" not in vector_context:
                # Split by KONTEKST markers and clean
                parts = vector_context.split("KONTEKST")
                for part in parts:
                    cleaned = part.strip()
                    # Remove numbering like "1:", "2:", etc.
                    if cleaned and len(cleaned) > 10:
                        if cleaned[0].isdigit() and ":" in cleaned[:3]:
                            cleaned = cleaned.split(":", 1)[-1].strip()
                        if cleaned:
                            context_list.append(cleaned)

        if use_graph and len(results) > 1:
            graph_context = results[1].response
            if graph_context and "Nie znaleziono" not in graph_context:
                context_list.append(graph_context)

        return context_list

    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Get summary statistics from evaluation results."""
        # Handle NaN values gracefully
        faithfulness_vals = df["faithfulness"].dropna()
        relevancy_vals = df["answer_relevancy"].dropna()
        precision_vals = df["context_precision"].dropna()

        stats = {
            "avg_faithfulness": (
                float(faithfulness_vals.mean()) if len(faithfulness_vals) > 0 else 0.0
            ),
            "avg_answer_relevancy": (
                float(relevancy_vals.mean()) if len(relevancy_vals) > 0 else 0.0
            ),
            "avg_context_precision": (
                float(precision_vals.mean()) if len(precision_vals) > 0 else 0.0
            ),
            "total_questions": len(df),
            "successful_evals": len(faithfulness_vals),
            "failed_evals": len(df) - len(faithfulness_vals),
            "min_faithfulness": (
                float(faithfulness_vals.min()) if len(faithfulness_vals) > 0 else 0.0
            ),
            "max_faithfulness": (
                float(faithfulness_vals.max()) if len(faithfulness_vals) > 0 else 0.0
            ),
        }

        self._logger.info(f"Summary stats: {stats}")
        return stats
