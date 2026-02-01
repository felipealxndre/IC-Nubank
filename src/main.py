from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from nodes_from_chunks import load_nodes_from_chunks

# importing retrievers
# from retrievers.hybrid import Hybrid
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
# importing agents
from agents import StandardAgent

load_dotenv()

def main():

    root_dir = Path(__file__).resolve().parents[1]
    chunks_path = root_dir / "data" / "processed" / "chunks.jsonl"

    # fluxo de RAG
    query = "Quero um estudo sobre as competências da BNCC relacionadas à triângulo retângulo."

    
    # para a geração de resposta após recuperação + rewriting de queries
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    # carregando nodes que estão salvos em chunks.jsonl
    nodes = load_nodes_from_chunks(path= chunks_path)
    
    # escolhendo um retriever - Que tipo de indexação usar

    # bm25 é um retriever baseado em palavras-chave(lexical)
    bm25 = BM25Retriever(
        nodes=nodes,
        persist_dir = root_dir / "indexes" / "bm25", # indices salvos
        top_k = 5,
    )
    # dense é um retriever baseado em embeddings (vetorial)
    dense = DenseRetriever(
        nodes=nodes,
        persist_dir = root_dir / "indexes" / "dense", # indices salvos
        top_k = 5,
    )

    # hybrid combina os dois retrievers acima
    # hybrid = Hybrid(
    #     retrievers = [bm25, dense],
    #     rrf_k = 10,
    #     top_k=5,
    # )

    # retrievers = dense, hybrid, bm25 
    retriever = bm25

    # escolhendo o agent - RAG Standard or RAG-f (Fusion)
    
    # RAG-Fusion
    #rewriter = QueryRewriter(llm=llm, n = 3)
    #agent = FusionAgent(retriever=retriever, rewriter=rewriter)

    # Standard RAG
    agent = StandardAgent(retriever=retriever, top_k=5)
    
    
    # recuperando as informações
    top_k_results = agent.retrieve(query=query)


    # apenas printando os top k resultados
    for i, item in enumerate(top_k_results, start=1):
        # metadata
        meta = item.node.metadata or {}
        page = meta.get("page_label") or meta.get("page") or "na"
        file_name = meta.get("file_name") or meta.get("filename") or "unknown"

        # id e score
        chunk_id = item.node.node_id
        score = float(item.score) if item.score is not None else 0.0

        # texto (uma linha + truncado)
        text = (item.node.text or "").strip().replace("\n", " ")
        text = " ".join(text.split())  # remove múltiplos espaços
        preview = (text[:220] + "…") if len(text) > 220 else text

        print(f"\n[{i}/{len(top_k_results)}] score={score:.4f} | page={page} | file={file_name}")
        print(f"chunk_id: {chunk_id}")
        print(f"preview : {preview}")

if __name__ == "__main__":
    main()