import json
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from backend.app.db.models import Document, ParentChunk

class KnowledgeGraphBuilder:
    """
    Constructs an interactive Knowledge Graph representation from indexed documents,
    tags, cross-references, and structural hierarchies.
    """

    @staticmethod
    async def build_graph(session: AsyncSession) -> Dict[str, Any]:
        stmt = select(Document)
        res = await session.execute(stmt)
        docs = res.scalars().all()

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_nodes = set()
        seen_edges = set()

        tag_counts: Dict[str, int] = {}

        # 1. Add Document Nodes
        for doc in docs:
            doc_node_id = f"doc_{doc.id}"
            if doc_node_id not in seen_nodes:
                nodes.append({
                    "id": doc_node_id,
                    "label": doc.title,
                    "title": f"Document: {doc.title}\nType: {doc.file_type}",
                    "group": "document",
                    "value": max(15, min(35, doc.byte_size // 150)),
                    "doc_id": doc.id,
                    "filename": doc.filename
                })
                seen_nodes.add(doc_node_id)

            # Process Tags
            try:
                tags = json.loads(doc.tags) if doc.tags else []
            except Exception:
                tags = []

            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                tag_node_id = f"tag_{tag}"
                if tag_node_id not in seen_nodes:
                    nodes.append({
                        "id": tag_node_id,
                        "label": f"#{tag}",
                        "title": f"Tag: #{tag}",
                        "group": "tag",
                        "value": 12
                    })
                    seen_nodes.add(tag_node_id)

                # Edge: Document -> Tag
                edge_id = f"{doc_node_id}->{tag_node_id}"
                if edge_id not in seen_edges:
                    edges.append({
                        "id": edge_id,
                        "from": doc_node_id,
                        "to": tag_node_id,
                        "label": "tagged",
                        "weight": 1.0
                    })
                    seen_edges.add(edge_id)

        # 2. Add Cross-Document References
        # Match wikilinks or title mentions in raw_content
        for doc in docs:
            doc_node_id = f"doc_{doc.id}"
            content_lower = doc.raw_content.lower()

            for other_doc in docs:
                if doc.id == other_doc.id:
                    continue

                other_node_id = f"doc_{other_doc.id}"
                # Check if other doc title is mentioned in this doc
                if other_doc.title.lower() in content_lower:
                    edge_id = f"{doc_node_id}->{other_node_id}"
                    rev_edge_id = f"{other_node_id}->{doc_node_id}"
                    if edge_id not in seen_edges and rev_edge_id not in seen_edges:
                        edges.append({
                            "id": edge_id,
                            "from": doc_node_id,
                            "to": other_node_id,
                            "label": "references",
                            "weight": 2.0
                        })
                        seen_edges.add(edge_id)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_documents": len(docs),
                "total_tags": len(tag_counts),
                "total_connections": len(edges)
            }
        }
