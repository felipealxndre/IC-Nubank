from pathlib import Path
from llama_index.core.schema import TextNode
import json

def load_nodes_from_chunks(path: Path, text_field: str = "text_raw") -> list[TextNode]:
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
            meta = dict(chunk.get("metadata", {}) or {})
    
            meta["text_raw"] = chunk.get("text_raw", "")
            meta["text_lex"] = chunk.get("text_lex", "")

            node = TextNode(text=chunk.get(text_field, ""), metadata=meta)
            node.node_id = chunk["chunk_id"]

            node.excluded_embed_metadata_keys = list(meta.keys())
            nodes.append(node)
    return nodes