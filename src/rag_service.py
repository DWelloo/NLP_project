from src.llm import BaseLlm
from src.models import RagResult
from src.logger import get_logger
from src.graph_service import GraphService
from src.vector_service import VectorService


class RAGService:
    def __init__(self, vector: VectorService, graph: GraphService, llm: BaseLlm):
        self._vector = vector
        self._graph = graph
        self._llm = llm
        self._logger = get_logger("RAGService")

    def ingest_vector(self, file_path: str):
        self._vector.ingest_document(file_path)

    def ingest_graph(self, file_path: str):
        self._graph.ingest_document(file_path)


    def route_and_respond(self, query: str, use_vector: bool, use_graph: bool) -> list[RagResult]:
        if use_vector:
            vector_results = self._vector.search(query, k=3)
            vector_context = ""
            for i in range(len(vector_results)):
                vector_context += f"KONTEKST {i+1}:\n{vector_results[i].page_content}\n\n"
        else:
            vector_context = "Brak użycia bazy wektorowej."

        if use_graph:
            graph_results = self._graph.search(query)
        else:
            graph_results = "Brak użycia bazy grafowej."
        

        hybrid_prompt = f"""
        Jesteś asystentem integrującym dane z dwóch systemów: grafu wiedzy i bazy wektorowej.
        
        TWOJE ZADANIE: Stwórz podsumowanie, które łączy fakty z grafu oraz kontekst z tekstów.
        
        DANE Z GRAFU (Fakty):
        {graph_results};

        DANE WEKTOROWE (Kontekst):
        {vector_context};
        
        ZAPYTANIE UŻYTKOWNIKA: '{query}'.
        """
        
        self._logger.debug(f"Hybrid prompt for LLM:\n{hybrid_prompt}")
        hybrid_answer = self._llm.generate(hybrid_prompt)

        return [
            RagResult("Baza wektorowa", vector_context if vector_context else "Brak pasujących fragmentów."),
            RagResult("Baza grafowa", graph_results),
            RagResult("Fuzja odpowiedzi z obu baz, wygenerowana przez LLM", hybrid_answer)
        ]