import json
import re
from typing import Dict, List, Optional
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters.base import TextSplitter
from llama_index.core.schema import TextNode
from src.relation_extraction import GeminiExtractor
from src.file_loaders import get_loader
from src.logger import get_logger
from src.query_processor import get_query_processor
from src.llm import BaseLlm


class GraphService:
    def __init__(
        self,
        neo4j: Neo4jGraph,
        splitter: TextSplitter,
        extractor: GeminiExtractor,
        llm: BaseLlm,
        clear_db: bool = False,
    ):
        self._logger = get_logger("GraphService")
        self._neo4j = neo4j
        self._splitter = splitter
        self._extractor = extractor
        self._llm = llm
        self._query_processor = get_query_processor()
        self._community_graph_name = "community_graph"
        if clear_db:
            self.clear_db()

    def clear_db(self):
        self._logger.debug("Clearing Neo4j database...")
        self._neo4j.query("MATCH (n) DETACH DELETE n")

    def ingest_document(self, file_path: str):
        self._logger.debug(f"Splitting document {file_path}...")
        loader = get_loader(file_path)
        langchain_docs = self._splitter.split_documents(loader.load())
        nodes = [
            TextNode(text=doc.page_content, metadata=doc.metadata, id_=f"node_{i}")
            for i, doc in enumerate(langchain_docs)
        ]

        relations = self._extractor.extract_relations(nodes)
        self._logger.debug(
            f"Extracted {len(relations)} relations from document {file_path}"
        )

        # Store in Neo4j
        stored_count = 0
        for rel in relations:
            try:
                subject = self._normalize_name(rel["subject"])
                obj = self._normalize_name(rel["object"])
                predicate = self._normalize_name(rel["predicate"])

                self._insert_relation(subject, predicate, obj)
                stored_count += 1
            except Exception as e:
                self._logger.error(f"Error storing relation: {e}")
                continue

        self._logger.debug(f"Stored {stored_count} relations from {len(nodes)} nodes.")

        self._logger.debug("Refreshing Neo4j schema...")
        self._neo4j.refresh_schema()

    def _insert_relation(self, subject: str, predicate: str, obj: str):

        query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $obj})
            WITH s, o
            CALL apoc.create.relationship(s, $predicate, {}, o) YIELD rel
            RETURN rel
        """
        params = {"subject": subject, "obj": obj, "predicate": predicate}
        self._neo4j.query(query, params)

    def _normalize_name(self, s: str) -> str:
        if not s:
            return s

        s = s.strip().lower().replace("'", "\\'").replace('"', '\\"')
        return s

    def search(self, query: str) -> str:
        """Search the graph using lemmatization-based keyword extraction."""
        self._logger.debug(f"Performing graph search for: {query}")

        # Extract keywords using NLP-based processor
        keywords = self._query_processor.extract_keywords(query)

        if not keywords:
            self._logger.warning("No keywords extracted, using fallback")
            keywords = [query.lower()]

        self._logger.debug(f"Extracted keywords: {keywords}")

        all_results = []

        for keyword in keywords[:5]:  # Limit to 5 keywords
            # Get search variants (lemma + partial matches)
            variants = self._query_processor.get_search_variants(keyword)

            for variant in variants:
                search_query = """
                MATCH (n)-[r]->(m)
                WHERE toLower(n.name) CONTAINS toLower($keyword)
                   OR toLower(m.name) CONTAINS toLower($keyword)
                RETURN n.name as source, type(r) as relation, m.name as target
                LIMIT 5
                """

                try:
                    results = self._neo4j.query(
                        search_query, params={"keyword": variant}
                    )
                    all_results.extend(results)
                except Exception as e:
                    self._logger.warning(f"Search error for '{variant}': {e}")
                    continue

        if not all_results:
            return "Nie znaleziono powiązanych informacji w grafie wiedzy."

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        # Format results
        formatted = []
        for r in unique_results[:15]:  # Limit total results
            formatted.append(f"- {r['source']} --[{r['relation']}]--> {r['target']}")

        self._logger.debug(f"Found {len(unique_results)} unique relations")
        return "Znalezione relacje w grafie:\n" + "\n".join(formatted)

    def _prepare_gds_graph(self, graph_name: str = "community_graph"):
        self._logger.debug(f"Preparing GDS graph projection '{graph_name}'...")

        self._neo4j.query(f"CALL gds.graph.drop('{graph_name}', false)")

        query = f"""
        CALL gds.graph.project(
            '{graph_name}',
            'Entity',
            {{
                ALL_RELATIONS: {{
                    type: '*',
                    orientation: 'UNDIRECTED'
                }}
            }}
        )
        """
        self._neo4j.query(query)
        self._logger.debug(f"GDS graph projection '{graph_name}' created.")

    def _run_community_detection(
        self, graph_name: str = "community_graph", levels: int = 10
    ):
        self._logger.debug("Running community detection using Leiden algorithm...")
        query = f"""
        CALL gds.leiden.write('{graph_name}', {{
            writeProperty: 'communityId',
            maxLevels: {levels}
        }})
        YIELD communityCount, modularity
        """
        result = self._neo4j.query(query)
        self._logger.debug(f"Detected {result[0]['communityCount']} communities.")
        return result[0]

    def _summarize_communities(self, batch_size: int = 10):
        self._logger.debug("Fetching communities for summarization...")

        query = """
            MATCH (n:Entity)
            WHERE n.communityId IS NOT NULL
            RETURN n.communityId AS id, collect(n.name)[0..30] AS nodes
        """
        communities = self._neo4j.query(query)

        if not communities:
            self._logger.warning("No communities found to summarize.")
            return {}

        all_summaries = {}

        for i in range(0, len(communities), batch_size):
            batch = communities[i : i + batch_size]
            self._logger.debug(
                f"Processing batch {i//batch_size + 1} ({len(batch)} communities)..."
            )

            prompt = "Podsumuj krótko (max 2 zdania) każdą z poniższych grup encji z grafu wiedzy:\n\n"
            for comm in batch:
                prompt += f"GRUPA_ID {comm['id']}: {', '.join(comm['nodes'])}\n"

            prompt += '\nZwróć odpowiedź w formacie JSON: {"ID": "Podsumowanie", ...}'

            try:
                response = self._llm.generate(prompt)

                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    batch_summaries = json.loads(match.group(0))

                    for comm_id_str, summary in batch_summaries.items():
                        comm_id = int(comm_id_str.replace("GRUPA_ID ", ""))
                        all_summaries[comm_id] = summary
                        self._store_community_summary(comm_id, summary)

            except Exception as e:
                self._logger.error(f"Error in batch summarization: {e}")
                continue

        return all_summaries

    def _store_community_summary(self, community_id: int, summary: str):
        query = """
            MATCH (n:Entity {communityId: $id})
            SET n.communitySummary = $summary
        """
        self._neo4j.query(query, {"id": community_id, "summary": summary})

    def build_communities(self, levels: int = 10, batch_size: int = 10):
        self._logger.debug(
            f"Starting community detection process on {levels} levels..."
        )
        self._prepare_gds_graph(self._community_graph_name)
        stats = self._run_community_detection(self._community_graph_name, levels)
        self._summarize_communities(batch_size)
        return stats
