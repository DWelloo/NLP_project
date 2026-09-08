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

from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.logger import get_logger
from src.rag_service import RAGService
from src.config import GEMINI_API_KEY, EMBEDDINGS_MODEL, GEMINI_MODEL_NAME, OPENAI_KEY
from langchain_openai.chat_models import ChatOpenAI as EvalLLM


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

    # Evaluation model name (can be set via GEMINI_MODEL_NAME env var)
    EVAL_MODEL_NAME = "gpt-4o-mini"

    # Configuration for API
    REQUEST_TIMEOUT = 120
    MAX_RETRIES = 3
    MAX_WORKERS = 1  # Single worker to minimize rate limiting

    def __init__(self, rag_service: RAGService):
        self._logger = get_logger("EvaluationService")
        self._rag = rag_service

        # Initialize ChatOpenAI (GPT) via LangChain for RAGAS
        self._logger.info(
            f"Initializing ChatOpenAI LLM for RAGAS (model={self.EVAL_MODEL_NAME})..."
        )
        langchain_llm = ChatOpenAI(
            model_name=self.EVAL_MODEL_NAME,
            openai_api_key=OPENAI_KEY,
            temperature=0,
            request_timeout=self.REQUEST_TIMEOUT,
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
        modes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Run RAGAS evaluation on the test dataset.

        Args:
            use_vector: Whether to use vector search (used only if modes is None)
            use_graph: Whether to use graph search (used only if modes is None)
            limit: Optional limit on number of questions to evaluate
            modes: Optional list of modes to evaluate. Allowed values: 'vector','graph','hybrid'.
                   If None, behavior falls back to previous single-run logic derived from
                   use_vector/use_graph.

        Returns:
            DataFrame with evaluation results (includes 'approach' column)
        """
        self._logger.info("=" * 50)
        self._logger.info(f"Starting RAGAS evaluation with {self.EVAL_MODEL_NAME}...")
        self._logger.info(
            f"Settings: use_vector={use_vector}, use_graph={use_graph}, limit={limit}, modes={modes}"
        )
        self._logger.info("=" * 50)

        test_data = self.load_test_dataset()
        if limit:
            test_data = test_data[:limit]

        # Determine modes to run
        allowed = {"vector", "graph", "hybrid"}
        if modes is None:
            # fallback to single mode to preserve prior behavior
            if use_vector and use_graph:
                modes = ["hybrid"]
            elif use_vector:
                modes = ["vector"]
            elif use_graph:
                modes = ["graph"]
            else:
                raise ValueError("At least one of use_vector/use_graph must be True")
        else:
            # validate provided modes
            modes = [m.lower() for m in modes]
            for m in modes:
                if m not in allowed:
                    raise ValueError(f"Invalid mode '{m}'. Allowed: {allowed}")

        self._logger.info(
            f"Evaluating {len(test_data)} questions across modes: {modes}"
        )

        # Collect RAG responses for each question+mode
        questions = []
        ground_truths = []
        answers = []
        contexts = []
        approaches = []

        eval_llm = EvalLLM(
            model_name=self.EVAL_MODEL_NAME,
            openai_api_key=OPENAI_KEY,
            temperature=0,
        )

        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            self._logger.info(f"[{i+1}/{len(test_data)}] Queueing: {question[:50]}...")

            for mode in modes:
                use_v = mode in ("vector", "hybrid")
                use_g = mode in ("graph", "hybrid")

                self._logger.debug(
                    f"  -> Mode '{mode}': use_vector={use_v}, use_graph={use_g}"
                )

                try:
                    results = self._rag.route_and_respond(
                        question, use_vector=use_v, use_graph=use_g
                    )

                    # Extract answer (last result assumed hybrid/final)
                    generated_answer = str(results[-1].response) if results else ""
                    self._logger.debug(
                        f"    -> Answer length: {len(generated_answer)} chars"
                    )

                    # Extract contexts
                    context_list = self._extract_contexts(results, use_v, use_g)
                    self._logger.debug(f"    -> Contexts found: {len(context_list)}")

                    questions.append(question)
                    ground_truths.append(ground_truth)
                    answers.append(generated_answer)
                    contexts.append(
                        context_list if context_list else ["Brak dostępnego kontekstu"]
                    )
                    approaches.append(mode)

                except Exception as e:
                    self._logger.error(
                        f"    -> Error processing question (mode={mode}): {e}"
                    )
                    questions.append(question)
                    ground_truths.append(ground_truth)
                    answers.append(f"Error: {str(e)}")
                    contexts.append(["Błąd przetwarzania"])
                    approaches.append(mode)

        total_evals = len(test_data) * len(modes)
        self._logger.info(f"Total prepared evaluations: {total_evals}")

        # Create RAGAS dataset
        self._logger.info("Creating RAGAS dataset...")
        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "ground_truth": ground_truths,
                "answer": answers,
                "contexts": contexts,
                "approach": approaches,
            }
        )

        self._logger.info("=" * 50)
        self._logger.info(f"Running RAGAS metrics with {self.EVAL_MODEL_NAME}...")
        self._logger.info(f"Total evaluations: {total_evals} (metrics per row)")
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
                raise_exceptions=False,
            )

            df = results.to_pandas()
            # ensure columns from raw data are present and ordered
            df.insert(0, "question", questions)
            df.insert(1, "ground_truth", ground_truths)
            df.insert(2, "answer", answers)
            df.insert(3, "approach", approaches)

            self._logger.info("=" * 50)
            self._logger.info("Evaluation complete!")
            self._logger.info(f"Results shape: {df.shape}")
            self._logger.info("=" * 50)

            return df

        except Exception as e:
            self._logger.error(f"RAGAS evaluation failed: {e}")
            import traceback

            self._logger.error(traceback.format_exc())
            out_df = pd.DataFrame(
                {
                    "question": questions,
                    "ground_truth": ground_truths,
                    "answer": answers,
                    "approach": approaches,
                    "faithfulness": [None] * len(questions),
                    "answer_relevancy": [None] * len(questions),
                    "context_precision": [None] * len(questions),
                }
            )
            out_df.to_csv("evaluation_output.csv", index=False)
            return out_df

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
