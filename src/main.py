from pathlib import Path

from llama_index.llms.openai import OpenAI
from nodes_from_chunks import load_nodes_from_chunks

# importing retrievers
from retrievers.hybrid import Hybrid
from retrievers.bm25 import BM25
from retrievers.dense import Dense
# importing agents
from agents.standard_agent import StandardAgent
from agents.fusion_agent import FusionAgent



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
    bm25 = BM25(
        nodes=nodes,
        persist_dir = root_dir / "indexes" / "bm25", # indices salvos
        top_k = 5,
    )
    # dense é um retriever baseado em embeddings (vetorial)
    dense = Dense(
        nodes=nodes,
        persist_dir = root_dir / "indexes" / "dense", # indices salvos
        top_k = 5,
    )

    # hybrid combina os dois retrievers acima
    hybrid = Hybrid(
        retrievers = [bm25, dense],
        rrf_k = 10,
        top_k=5,
    )

    # retriever = Dense()
    retriever = Hybrid()

    # escolhendo o agent - RAG Standard or RAG-f (Fusion)
    
    # RAG-Fusion
    rewriter = QueryRewriter(llm=llm, n = 3)
    agent = FusionAgent(retriever=retriever, rewriter=rewriter)

    agent = FusionAgent(llm=llm, retriever=retriever)

    # Standard RAG
    agent = StandardAgent(retriever=retriever)
    
    
    # recuperando as informações
    top_k_results = agent.retrieve(query=query)
    print("Top K Results:")
    print(top_k_results)

if __name__ == "__main__":
    main()