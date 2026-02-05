from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from nodes_from_chunks import load_nodes_from_chunks

# importing retrievers
from retrievers.hybrid import Hybrid
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
# importing agents
from agents import StandardAgent, FusionAgent
from query_rewrite import QueryRewriter
import json

from metrics import recall, mean_reciprocal_rank, normalized_discounted_cumulative_gain
from utils.reporting import generate_results

root_dir = Path(__file__).resolve().parents[1]

load_dotenv()

def main():

    root_dir = Path(__file__).resolve().parents[1]
    chunks_path = root_dir / "data" / "processed" / "chunks.jsonl"
    bench_path = root_dir / "bench" / "queries.json"
    top_k = 5

    with bench_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    rewriter = QueryRewriter(n=3)
    # escolhendo um retriever
    # bm25 é um retriever baseado em palavras-chave(lexical)
    bm25 = BM25Retriever(
        nodes=load_nodes_from_chunks(path= chunks_path, text_field="text_lex"),
        persist_dir = root_dir / "indexes" / "bm25", # indices salvos
        top_k = top_k,
    )
    # dense é um retriever baseado em embeddings (vetorial)
    dense = DenseRetriever(
        nodes=load_nodes_from_chunks(path= chunks_path, text_field="text_raw"),
        persist_dir = root_dir / "indexes" / "dense", # indices salvos
        top_k = top_k,
    )
    # hybrid combina os dois retrievers acima
    hybrid = Hybrid(retrievers = [bm25, dense], top_k=top_k)
    retrievers = [dense, bm25, hybrid]
    

    results = []
    for item in benchmark:
        query_id = item['id']
        query = item['query']
        relevant = item['relevant']
        for retriever in retrievers:
            # escolhendo o agent - RAG Standard or RAG-f (Fusion)
            # Standard RAG
            standard_rag = StandardAgent(retriever=retriever, top_k=top_k)
            # RAG-Fusion
            fusion_rag = FusionAgent(retriever=retriever, top_k=top_k, rewriter=rewriter)

            agents = [standard_rag, fusion_rag]
            for agent in agents:

                # recuperando as informações

                # Framework
                top_k_results = agent.retrieve(query=query)
                ranked_ids = [r.node.node_id for r in top_k_results]

                results.append({
                    "query_id": query_id,
                    "query": query,
                    "agent": agent.__class__.__name__,
                    "retriever": retriever.__class__.__name__,
                    f"recall@{top_k}": recall(ranked_ids, relevant, top_k),
                    f"mrr@{top_k}": mean_reciprocal_rank(ranked_ids, relevant, top_k),
                    f"ndcg@{top_k}": normalized_discounted_cumulative_gain(ranked_ids, relevant, top_k),
                })

                # apenas printando os top k resultados
                for i, item in enumerate(top_k_results, start=1):
                    # metadata
                    meta = item.node.metadata or {}
                    page = meta.get("page_label") or meta.get("page") or "na"
                    file_name = meta.get("file_name") or meta.get("filename") or "unknown"

                    # id e score
                    chunk_id = item.node.node_id
                    score = float(item.score) if item.score is not None else 0.0

                    # texto
                    raw = meta.get("text_raw") or item.node.text or ""
                    text = raw.strip().replace("\n", " ")
                    text = " ".join(text.split())
                    preview = (text[:220] + "…") if len(text) > 220 else text

                    print(f"\n[{i}/{len(top_k_results)}] score={score:.4f} | page={page} | file={file_name}")
                    print(f"chunk_id: {chunk_id}")
                    print(f"preview : {preview}")

    
    summary_rows, paths = generate_results(root_dir / "data" / "results", results, k=top_k)


if __name__ == "__main__":
    main()