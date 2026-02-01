from pathlib import Path
from llama_index.core.schema import TextNode
import json

def load_nodes_from_chunks(path: Path) -> list[TextNode]:
    """
    Lê data/processed/chunks.jsonl e retorna uma lista de TextNode.
    Cada TextNode recebe node_id = chunk_id
    """
    nodes: list[TextNode] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            node = TextNode(text=chunk["text"], metadata=chunk.get("metadata", {}))
            node.node_id = chunk["chunk_id"]
            nodes.append(node)
    return nodes