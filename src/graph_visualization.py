"""
Graph Visualization Module for Streamlit UI.

Provides interactive visualization of Neo4j knowledge graph using streamlit-agraph.
"""

from typing import Optional
from streamlit_agraph import agraph, Node, Edge, Config

from langchain_neo4j import Neo4jGraph
from src.logger import get_logger


class GraphVisualization:
    """Service for visualizing Neo4j graph data in Streamlit."""
    
    # Color palette for different node types
    NODE_COLORS = {
        "Disease": "#FF6B6B",      # Red
        "Symptom": "#4ECDC4",       # Teal
        "Drug": "#45B7D1",          # Blue
        "Treatment": "#96CEB4",     # Green
        "SideEffect": "#FFEAA7",    # Yellow
        "UNKNOWN": "#DDA0DD"        # Plum
    }
    
    def __init__(self, neo4j: Neo4jGraph):
        self._logger = get_logger("GraphVisualization")
        self._neo4j = neo4j
    
    def get_all_nodes_and_relations(self, limit: int = 100) -> tuple[list[dict], list[dict]]:
        """Get all nodes and relationships from the graph."""
        self._logger.debug(f"Fetching graph data (limit={limit})...")
        
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT {limit}
        """
        
        results = self._neo4j.query(query)
        
        nodes_dict = {}
        relationships = []
        
        for record in results:
            source = record.get("n", {})
            source_name = source.get("name", "Unknown")
            source_type = source.get("type", "UNKNOWN")
            if source_name not in nodes_dict:
                nodes_dict[source_name] = {"name": source_name, "type": source_type}
            
            target = record.get("m", {})
            target_name = target.get("name", "Unknown")
            target_type = target.get("type", "UNKNOWN")
            if target_name not in nodes_dict:
                nodes_dict[target_name] = {"name": target_name, "type": target_type}
            
            rel = record.get("r")
            if rel:
                rel_type = type(rel).__name__ if hasattr(rel, '__class__') else str(rel)
                if hasattr(rel, 'type'):
                    rel_type = rel.type
                elif isinstance(rel, tuple) and len(rel) > 1:
                    rel_type = rel[1]
                
                relationships.append({
                    "source": source_name,
                    "target": target_name,
                    "label": rel_type
                })
        
        return list(nodes_dict.values()), relationships
    
    def get_subgraph_for_query(self, search_term: str, depth: int = 2, limit: int = 50) -> tuple[list[dict], list[dict]]:
        """Get a subgraph related to a search term."""
        self._logger.debug(f"Searching for subgraph: '{search_term}' (depth={depth})")
        
        query = f"""
        MATCH path = (start)-[*1..{depth}]-(connected)
        WHERE toLower(start.name) CONTAINS toLower($search_term)
        WITH nodes(path) as nodes, relationships(path) as rels
        UNWIND nodes as n
        WITH DISTINCT n, rels
        UNWIND rels as r
        WITH n, DISTINCT r
        RETURN COLLECT(DISTINCT n) as nodes, COLLECT(DISTINCT r) as relationships
        LIMIT {limit}
        """
        
        try:
            results = self._neo4j.query(query, params={"search_term": search_term})
        except Exception as e:
            self._logger.warning(f"Complex query failed, trying simple search: {e}")
            return self._simple_search(search_term, limit)
        
        if not results or not results[0]:
            return self._simple_search(search_term, limit)
        
        return self._parse_results(results)
    
    def _simple_search(self, search_term: str, limit: int) -> tuple[list[dict], list[dict]]:
        """Simple fallback search for nodes matching the term."""
        query = f"""
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) CONTAINS toLower($search_term)
           OR toLower(m.name) CONTAINS toLower($search_term)
        RETURN n, r, m
        LIMIT {limit}
        """
        
        results = self._neo4j.query(query, params={"search_term": search_term})
        
        nodes_dict = {}
        relationships = []
        
        for record in results:
            source = record.get("n", {})
            source_name = source.get("name", "Unknown")
            source_type = source.get("type", "UNKNOWN")
            nodes_dict[source_name] = {"name": source_name, "type": source_type}
            
            target = record.get("m", {})
            target_name = target.get("name", "Unknown")
            target_type = target.get("type", "UNKNOWN")
            nodes_dict[target_name] = {"name": target_name, "type": target_type}
            
            rel = record.get("r")
            if rel:
                rel_type = getattr(rel, 'type', str(rel)) if hasattr(rel, 'type') else "RELATED"
                relationships.append({
                    "source": source_name,
                    "target": target_name,
                    "label": rel_type
                })
        
        return list(nodes_dict.values()), relationships
    
    def _parse_results(self, results: list) -> tuple[list[dict], list[dict]]:
        """Parse Neo4j results into nodes and relationships."""
        if not results:
            return [], []
        
        record = results[0]
        nodes_data = record.get("nodes", [])
        rels_data = record.get("relationships", [])
        
        nodes = []
        for n in nodes_data:
            if isinstance(n, dict):
                nodes.append({
                    "name": n.get("name", "Unknown"),
                    "type": n.get("type", "UNKNOWN")
                })
        
        relationships = []
        for r in rels_data:
            if hasattr(r, 'start_node') and hasattr(r, 'end_node'):
                relationships.append({
                    "source": r.start_node.get("name", "Unknown"),
                    "target": r.end_node.get("name", "Unknown"),
                    "label": getattr(r, 'type', "RELATED")
                })
        
        return nodes, relationships
    
    def render_graph(
        self, 
        nodes_data: list[dict], 
        relationships: list[dict],
        height: int = 500,
        physics: bool = True
    ) -> Optional[str]:
        """Render graph visualization using streamlit-agraph."""
        if not nodes_data:
            return None
        
        nodes = []
        for node in nodes_data:
            node_type = node.get("type", "UNKNOWN")
            color = self.NODE_COLORS.get(node_type, self.NODE_COLORS["UNKNOWN"])
            
            nodes.append(Node(
                id=node["name"],
                label=node["name"],
                size=25,
                color=color,
                title=f"Typ: {node_type}"
            ))
        
        edges = []
        for rel in relationships:
            edges.append(Edge(
                source=rel["source"],
                target=rel["target"],
                label=rel.get("label", ""),
                color="#888888"
            ))
        
        config = Config(
            width="100%",
            height=height,
            directed=True,
            physics=physics,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": True}
        )
        
        return agraph(nodes=nodes, edges=edges, config=config)
    
    def get_graph_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        node_count_query = "MATCH (n) RETURN count(n) as count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        type_query = "MATCH (n) RETURN DISTINCT n.type as type, count(*) as count ORDER BY count DESC"
        
        try:
            node_count = self._neo4j.query(node_count_query)[0]["count"]
            rel_count = self._neo4j.query(rel_count_query)[0]["count"]
            types = self._neo4j.query(type_query)
            
            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "node_types": {t["type"]: t["count"] for t in types if t["type"]}
            }
        except Exception as e:
            self._logger.error(f"Error getting graph stats: {e}")
            return {"total_nodes": 0, "total_relationships": 0, "node_types": {}}
