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

load_dotenv()

def main():

    root_dir = Path(__file__).resolve().parents[1]
    chunks_path = root_dir / "data" / "processed" / "chunks.jsonl"

    # fluxo de RAG
    queries = [
        "EM13MAT303",
        "Quais habilidades da BNCC envolvem teorema de Pitágoras no 9º ano?",
        "BNCC: relações métricas no triângulo retângulo (EF09MA13, EF09MA14).",
        "Geometria no Ensino Fundamental: triângulos e construção com régua e compasso.",
        "Quais são as competências/habilidades de Matemática sobre polígonos regulares?",
    ]

    # query = "Quais habilidades da BNCC envolvem teorema de Pitágoras?"

    
    # para a geração de resposta após recuperação + rewriting de queries
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # escolhendo um retriever - Que tipo de indexação usar

    # bm25 é um retriever baseado em palavras-chave(lexical)
    bm25 = BM25Retriever(
        nodes=load_nodes_from_chunks(path= chunks_path, text_field="text_lex"),
        persist_dir = root_dir / "indexes" / "bm25", # indices salvos
        top_k = 5,
    )
    # dense é um retriever baseado em embeddings (vetorial)
    dense = DenseRetriever(
        nodes=load_nodes_from_chunks(path= chunks_path, text_field="text_raw"),
        persist_dir = root_dir / "indexes" / "dense", # indices salvos
        top_k = 5,
    )

    # hybrid combina os dois retrievers acima
    hybrid = Hybrid(retrievers = [bm25, dense])

    # retrievers = dense, hybrid, bm25 
    retriever = hybrid

    # escolhendo o agent - RAG Standard or RAG-f (Fusion)
    
    # Standard RAG
    # agent = StandardAgent(retriever=retriever, top_k=5)
    
    
    # RAG-Fusion
    # agent = FusionAgent(retriever=retriever)

    for query in queries:

        print("#"*30 + f"{query}" + "#"*30 + "\n")

        for retriever in [dense, bm25, hybrid]:
            print("="*30 + f"{retriever.__class__.__name__}" + "="*30 + "\n")
            # recuperando as informações

            # Standard RAG
            agent = StandardAgent(retriever=retriever, top_k=5)

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

                # texto
                raw = meta.get("text_raw") or item.node.text or ""
                text = raw.strip().replace("\n", " ")
                text = " ".join(text.split())
                preview = (text[:220] + "…") if len(text) > 220 else text

                print(f"\n[{i}/{len(top_k_results)}] score={score:.4f} | page={page} | file={file_name}")
                print(f"chunk_id: {chunk_id}")
                print(f"preview : {preview}")

if __name__ == "__main__":
    main()