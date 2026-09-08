import os
import streamlit as st

from src.llm import GeminiLlm
from src.chunking_strategy import ChunkingStrategy
from src.config import (
    DATA_PATH,
    EMBEDDINGS_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    NEO4J_PASS,
    NEO4J_URI,
    NEO4J_USER,
    QDRANT_URL,
    VECTOR_COLLECTION_NAME,
)
from src.graph_service import GraphService
from src.rag_service import RAGService
from src.vector_service import VectorService
from src.graph_visualization import GraphVisualization
from src.evaluation_service import EvaluationService
from src.relation_extraction import GeminiExtractor
from src.chunking import get_splitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph


@st.cache_resource
def get_neo4j() -> Neo4jGraph:
    """Get Neo4j connection (standalone, without full RAG init)."""
    return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Get embeddings model (standalone)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


@st.cache_resource
def get_llm() -> GeminiLlm:
    """Get LLM (standalone)."""
    return GeminiLlm(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL_NAME)


default_chunking_strategy = [ChunkingStrategy.FIXED, 2000, 200]


@st.cache_resource
def get_rag() -> RAGService:
    with st.spinner("Inicjalizacja systemu RAG..."):
        os.makedirs(DATA_PATH, exist_ok=True)
        st.session_state.last_response = []

        with st.spinner("Inicjalizacja modelu LLM..."):
            llm = get_llm()

        embeddings = get_embeddings()
        neo4j = get_neo4j()

        with st.spinner("Inicjalizacja bazy wektorowej..."):
            vector_service = VectorService(
                QDRANT_URL, VECTOR_COLLECTION_NAME, embeddings
            )

        with st.spinner("Konfiguracja bazy wektorowej..."):
            vector_service.set_chunking_strategy(
                chunking_strategy=default_chunking_strategy[0],
                chunk_size=default_chunking_strategy[1],
                chunk_overlap=default_chunking_strategy[2],
            )

        with st.spinner("Inicjalizacja bazy grafowej..."):
            gemini_extractor = GeminiExtractor(api_key=GEMINI_API_KEY)
            graph_splitter = get_splitter(
                strategy=default_chunking_strategy[0],
                chunk_size=default_chunking_strategy[1],
                chunk_overlap=default_chunking_strategy[2],
            )
            graph_service = GraphService(neo4j, graph_splitter, gemini_extractor, llm)

        with st.spinner("Inicjalizacja routera RAG..."):
            return RAGService(vector_service, graph_service, llm)


def run_streamlit():
    st.set_page_config(page_title="Hybrydowy System RAG", layout="wide")

    # Create tabs for different functionalities
    tab_chat, tab_graph, tab_eval = st.tabs(
        ["Czat RAG", "Wizualizacja Grafu", "Ewaluacja"]
    )

    with tab_chat:
        chat_tab()

    with tab_graph:
        graph_visualization_tab()

    with tab_eval:
        evaluation_tab()


def chat_tab():
    """Main chat and indexing functionality."""
    st.title("Hybrydowy System RAG")
    rag = get_rag()
    st.divider()

    # routing flag for disabling UI while LLM decides
    if "routing_in_progress" not in st.session_state:
        st.session_state["routing_in_progress"] = False

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baza Wektorowa (Qdrant)")

        if st.button("Wyczyść bazę wektorową", type="secondary"):
            rag._vector.clear_db()
            st.toast("Baza wektorowa została wyczyszczona!")

        st.divider()

        chunking_config = rag._vector.current_chunking_config
        st.text(
            f"Chunkowanie dokumentów: {chunking_config[0].value.capitalize()} (Size: {chunking_config[1]}, Overlap: {chunking_config[2]})"
        )

        chunking_strategy = st.selectbox(
            "Strategia chunkowania",
            options=list(ChunkingStrategy),
            format_func=lambda x: x.value.capitalize(),
            index=[e.name for e in ChunkingStrategy].index(
                chunking_config[0].name.upper()
            ),
        )

        chunking_size = None
        chunking_overlap = None
        if chunking_strategy != ChunkingStrategy.SEMANTIC:
            chunking_size = st.number_input("Rozmiar", value=chunking_config[1])
            chunking_overlap = st.number_input("Overlap", value=chunking_config[2])

        if st.button("Zaktualizuj strategię chunkowania", type="secondary"):
            rag._vector.set_chunking_strategy(
                chunking_strategy=chunking_strategy,
                chunk_size=chunking_size,
                chunk_overlap=chunking_overlap,
            )
            st.toast("Strategia chunkowania została zaktualizowana!")
            st.rerun()

    with col2:
        st.subheader("Baza Grafowa (Neo4j)")
        if st.button("Wyczyść bazę grafową", type="secondary"):
            rag._graph.clear_db()
            st.toast("Baza grafowa została wyczyszczona!")

        st.divider()
        st.text("Community detection")
        levels = st.number_input(
            "Liczba poziomów społeczności", min_value=1, max_value=100, value=10, step=1
        )
        batch_size = st.number_input(
            "Maksymalna liczba węzłów wysyłana do LLM w jednym zapytaniu",
            min_value=1,
            value=10,
            step=10,
        )
        if st.button("Zbuduj społeczności w grafie", type="secondary"):
            with st.spinner("Trwa budowanie społeczności..."):
                rag._graph.build_communities(levels, batch_size)
            st.toast("Budowanie społeczności zakończone pomyślnie!")

    if rag is None:
        st.info("Skonfiguruj system RAG, aby kontynuować.")
        return

    st.divider()
    st.subheader("Dodaj nowe pliki")
    uploaded_file = st.file_uploader("Wybierz plik", type=["pdf", "txt", "docx", "md"])

    col1, col2 = st.columns(2)
    with col1:
        index_vector = st.checkbox(
            "Indeksuj do bazy wektorowej", value=True, disabled=uploaded_file is None
        )
    with col2:
        index_graph = st.checkbox(
            "Indeksuj do bazy grafowej", value=True, disabled=uploaded_file is None
        )

    if uploaded_file:
        if st.button("Uruchom indeksowanie", type="primary"):
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Trwa indeksowanie..."):
                if index_vector:
                    try:
                        rag.ingest_vector(file_path)
                    except Exception as e:
                        st.error("Błąd podczas indeksowania do bazy wektorowej")
                        raise e
                    st.toast("Indeksowanie do bazy wektorowej zakończone pomyślnie!")
                if index_graph:
                    try:
                        rag.ingest_graph(file_path)
                    except Exception as e:
                        st.error("Błąd podczas indeksowania do bazy grafowej")
                        raise e
                    st.toast("Indeksowanie do bazy grafowej zakończone pomyślnie!")

    st.divider()
    query = st.text_input("Zadaj pytanie:")
    col3, col4 = st.columns(2)
    # checkboxes are disabled while LLM routing is in progress
    with col3:
        query_vector = st.checkbox(
            "Użyj bazy wektorowej",
            value=True,
            disabled=st.session_state.get("routing_in_progress", False),
            key="query_vector_checkbox",
        )
    with col4:
        query_graph = st.checkbox(
            "Użyj bazy grafowej",
            value=True,
            disabled=st.session_state.get("routing_in_progress", False),
            key="query_graph_checkbox",
        )

    # New: LLM-driven routing button
    if st.button(
        "LLM: wybierz źródło (VECTOR / GRAPH / BOTH)",
        type="secondary",
        disabled=(not query or st.session_state.get("routing_in_progress", False)),
    ):
        st.session_state["routing_in_progress"] = True
        try:
            with st.spinner("LLM wybiera źródła..."):
                llm = get_llm()
                prompt = """Jesteś agentem routującym zapytania.
Twoim zadaniem jest przeanalizować zapytanie użytkownika i zdecydować,
jakiego typu wyszukiwanie należy użyć:
1. VECTOR_SEARCH
   Użyj, gdy zapytanie jest:
   - opisowe lub ogólne
   - dotyczy znaczenia, wyjaśnień, definicji
   - nieprecyzyjne, „na sens”
   - typu: „co to jest”, „jak działa”, „wyjaśnij”
2. GRAPH_SEARCH
   Użyj, gdy zapytanie:
   - dotyczy konkretnych faktów lub encji
   - pyta o relacje, powiązania, zależności
   - wymaga struktury (kto, co z czym, kiedy)
3. BOTH
   Użyj, gdy zapytanie:
   - wymaga zrozumienia znaczenia
   ORAZ
   - sprawdzenia relacji lub faktów
Wypisz TYLKO jedną etykietę:
  VECTOR_SEARCH
  GRAPH_SEARCH
  BOTH
Zapytanie użytkownika:
"{user_query}"
                    """

                decision_text = None
                # try common method names on GeminiLlm until one works
                for method in (
                    "decide_routing",
                    "decide",
                    "generate",
                    "predict",
                    "chat",
                    "complete",
                    "call",
                ):
                    if hasattr(llm, method):
                        try:
                            res = getattr(llm, method)(prompt)
                            if isinstance(res, str):
                                decision_text = res
                            elif isinstance(res, dict) and "text" in res:
                                decision_text = res["text"]
                            elif hasattr(res, "text"):
                                decision_text = res.text
                            else:
                                decision_text = str(res)
                            break
                        except Exception:
                            continue

                # basic fallback if LLM call failed
                if not decision_text:
                    text = query.lower()
                    use_vector = any(
                        w in text
                        for w in ("dokument", "tekst", "treść", "semant", "semantic")
                    )
                    use_graph = any(
                        w in text
                        for w in ("graf", "relacja", "węzeł", "neo4j", "relacje")
                    )
                    if not (use_vector or use_graph):
                        use_vector = use_graph = True
                else:
                    d = decision_text.upper()
                    use_vector = "VECTOR" in d or "WEKT" in d or ("BOTH" in d and True)
                    use_graph = "GRAPH" in d or "GRAF" in d or ("BOTH" in d and True)
                    # ensure BOTH sets both
                    if "BOTH" in d:
                        use_vector = use_graph = True

                # run routing using RAG service
                st.session_state.last_response = rag.route_and_respond(
                    query, use_vector=use_vector, use_graph=use_graph
                )

        except Exception as e:
            st.error(f"Błąd podczas routingu: {e}")
        finally:
            st.session_state["routing_in_progress"] = False

    if st.button("Zadaj pytanie", type="primary") and (query_vector or query_graph):
        with st.spinner("Generowanie odpowiedzi..."):
            st.session_state.last_response = rag.route_and_respond(
                query, use_vector=query_vector, use_graph=query_graph
            )

    for r in st.session_state.get("last_response", []):
        st.markdown(f"##### {r.source}")
        st.write(r.response)
        st.divider()


def graph_visualization_tab():
    """Interactive graph visualization tab - works independently of RAG init."""
    st.subheader("Wizualizacja Grafu Wiedzy")

    # Get Neo4j connection directly (no RAG init needed)
    try:
        neo4j = get_neo4j()
    except Exception as e:
        st.error(f"Błąd połączenia z Neo4j: {e}")
        return

    graph_viz = GraphVisualization(neo4j)

    # Graph statistics
    stats = graph_viz.get_graph_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liczba węzłów", stats.get("total_nodes", 0))
    with col2:
        st.metric("Liczba relacji", stats.get("total_relationships", 0))
    with col3:
        node_types = stats.get("node_types", {})
        st.metric("Typy węzłów", len(node_types))

    # Node type legend
    if node_types:
        st.markdown("**Typy węzłów w grafie:**")
        cols = st.columns(min(len(node_types), 5))
        for i, (node_type, count) in enumerate(node_types.items()):
            color = graph_viz.NODE_COLORS.get(
                node_type, graph_viz.NODE_COLORS.get("UNKNOWN", "#cccccc")
            )
            cols[i % 5].markdown(
                f'<span style="background-color:{color};padding:2px 8px;border-radius:4px;">{node_type}</span>: {count}',
                unsafe_allow_html=True,
            )

    st.divider()

    # Search and visualization options
    col_search, col_opts = st.columns([3, 1])
    with col_search:
        search_term = st.text_input(
            "Szukaj węzłów (pozostaw puste dla całego grafu):", key="graph_search"
        )
    with col_opts:
        limit = st.slider("Limit relacji", 10, 200, 50)
        physics = st.checkbox("Symulacja fizyki", value=True)

    if st.button("Pokaż graf", type="primary"):
        with st.spinner("Ładowanie grafu..."):
            if search_term:
                nodes, relationships = graph_viz.get_subgraph_for_query(
                    search_term, limit=limit
                )
            else:
                nodes, relationships = graph_viz.get_all_nodes_and_relations(
                    limit=limit
                )

            if nodes:
                st.success(
                    f"Znaleziono {len(nodes)} węzłów i {len(relationships)} relacji."
                )
                selected = graph_viz.render_graph(
                    nodes, relationships, height=600, physics=physics
                )
                if selected:
                    st.info(f"Wybrany węzeł: {selected}")
            else:
                st.warning(
                    "Nie znaleziono węzłów. Upewnij się, że graf został zaindeksowany."
                )


def evaluation_tab():
    """RAGAS evaluation dashboard tab - supports vector, graph and hybrid approaches."""
    st.subheader("Ewaluacja Jakości Systemu RAG (RAGAS)")

    st.markdown(
        """
    Ewaluacja wykorzystuje framework **RAGAS** do oceny jakości odpowiedzi systemu RAG.
    
    **Metryki:**
    - **Faithfulness** - Czy odpowiedź jest zgodna z kontekstem (brak halucynacji)
    - **Answer Relevancy** - Czy odpowiedź jest trafna względem pytania
    - **Context Precision** - Czy pobrany kontekst jest precyzyjny
    """
    )

    st.divider()

    # Evaluation options: choose which approaches to run
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Wybierz podejścia do ewaluacji**")
        eval_vector = st.checkbox("Vector only", value=True, key="eval_vector_only")
        eval_graph = st.checkbox("Graph only", value=True, key="eval_graph_only")
        eval_hybrid = st.checkbox("Hybrid (both)", value=True, key="eval_hybrid")
    with col2:
        limit_questions = st.slider("Liczba pytań do ewaluacji", 1, 20, 1)

    if st.button("Uruchom ewaluację", type="primary"):
        with st.spinner("Inicjalizacja komponentów ewaluacji..."):
            try:
                rag = get_rag()
                eval_service = EvaluationService(rag_service=rag)
            except Exception as e:
                st.error(f"Błąd inicjalizacji: {e}")
                return

        modes = []
        if eval_vector:
            modes.append("vector")
        if eval_graph:
            modes.append("graph")
        if eval_hybrid:
            modes.append("hybrid")
        if not modes:
            # default to all if none selected
            modes = ["vector", "graph", "hybrid"]

        with st.spinner("Trwa ewaluacja... (może potrwać kilka minut)"):
            try:
                results_df = eval_service.run_evaluation(
                    modes=modes, limit=limit_questions
                )

                st.session_state.eval_results = results_df
                st.session_state.eval_stats = eval_service.get_summary_stats(results_df)
                st.session_state.eval_approach_modes = modes

                st.success("Ewaluacja zakończona!")

            except Exception as e:
                st.error(f"Błąd podczas ewaluacji: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

    # Display results if available
    if "eval_results" in st.session_state:
        st.divider()
        st.subheader("Wyniki Ewaluacji")

        stats = st.session_state.eval_stats
        modes = st.session_state.get(
            "eval_approach_modes", ["vector", "graph", "hybrid"]
        )
        df = st.session_state.eval_results.copy()

        # Overall summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Średnia Faithfulness",
                f"{stats['avg_faithfulness']:.2%}",
                help="Zgodność odpowiedzi z kontekstem",
            )
        with col2:
            st.metric(
                "Średnia Answer Relevancy",
                f"{stats['avg_answer_relevancy']:.2%}",
                help="Trafność odpowiedzi",
            )
        with col3:
            st.metric(
                "Średnia Context Precision",
                f"{stats['avg_context_precision']:.2%}",
                help="Precyzja kontekstu",
            )

        # Per-approach stats
        import pandas as pd

        agg = (
            df.groupby("approach")[
                ["faithfulness", "answer_relevancy", "context_precision"]
            ]
            .agg(["mean", "count"])
            .fillna(0)
        )

        st.divider()
        st.markdown("**Metryki wg podejścia**")
        cols = st.columns(len(modes))
        for i, mode in enumerate(modes):
            col = cols[i % len(cols)]
            if mode in agg.index:
                mean_f = agg.loc[mode, ("faithfulness", "mean")]
                mean_r = agg.loc[mode, ("answer_relevancy", "mean")]
                mean_c = agg.loc[mode, ("context_precision", "mean")]
                count = int(agg.loc[mode, ("faithfulness", "count")])
                col.metric(
                    f"{mode.capitalize()} — Faithfulness",
                    f"{mean_f:.2%}",
                    delta=f"N={count}",
                )
                col.metric(f"{mode.capitalize()} — Relevancy", f"{mean_r:.2%}")
                col.metric(f"{mode.capitalize()} — Context Precision", f"{mean_c:.2%}")
            else:
                col.markdown(f"**{mode.capitalize()}**")
                col.info("Brak wyników dla tego podejścia")

        # Detailed results table with approach filter
        st.divider()
        st.markdown("**Szczegółowe wyniki:**")

        approach_filter = st.selectbox(
            "Pokaż podejście",
            options=["all"] + sorted(df["approach"].unique().tolist()),
            index=0,
        )

        display_df = df.copy()
        if approach_filter != "all":
            display_df = display_df[display_df["approach"] == approach_filter]

        display_df = display_df[
            [
                "question",
                "approach",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
            ]
        ]
        display_df.columns = [
            "Pytanie",
            "Podejście",
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
        ]

        st.dataframe(
            display_df.style.background_gradient(
                subset=["Faithfulness", "Answer Relevancy", "Context Precision"],
                cmap="RdYlGn",
            ),
            use_container_width=True,
        )

        # Simple bar chart: average metric per approach
        st.divider()
        st.markdown("**Wizualizacja wyników wg podejścia:**")

        chart_df = (
            df.groupby("approach")[
                ["faithfulness", "answer_relevancy", "context_precision"]
            ]
            .mean()
            .rename(
                columns={
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy",
                    "context_precision": "Context Precision",
                }
            )
        )
        if not chart_df.empty:
            st.bar_chart(chart_df)
