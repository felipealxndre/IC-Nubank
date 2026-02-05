from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nodes_from_chunks import load_nodes_from_chunks

from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from retrievers.hybrid import Hybrid

from agents import StandardAgent, FusionAgent
from query_rewrite import QueryRewriter


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
QUERIES_PATH = ROOT_DIR / "bench" / "queries.json"

MODEL = "gpt-4o-mini"

TOP_K_PER_SYSTEM = 15
TOP_SAVE = 15

Score = Literal[0, 1, 2, 3]


class ChunkJudge(BaseModel):
    chunk_id: str = Field(..., description="ID do trecho avaliado.")
    score: Score = Field(..., description="0=irrelevante, 3=muito relevante.")
    rationale: Optional[str] = Field(None, description="Motivo curto (opcional).")


def one_line(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    return " ".join(text.split())


def preview(text: str, max_chars: int = 900) -> str:
    t = one_line(text)
    return (t[:max_chars] + "…") if len(t) > max_chars else t


def load_chunk_text_map(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("chunk_id")
            if not cid:
                continue
            m[cid] = c.get("text_raw") or c.get("text") or ""
    return m


def build_retrievers(top_k: int):
    nodes_lex = load_nodes_from_chunks(CHUNKS_PATH, text_field="text_lex")
    nodes_raw = load_nodes_from_chunks(CHUNKS_PATH, text_field="text_raw")

    bm25 = BM25Retriever(
        nodes=nodes_lex,
        persist_dir=ROOT_DIR / "indexes" / "bm25",
        top_k=top_k,
        top_n=50,
    )

    dense = DenseRetriever(
        nodes=nodes_raw,
        persist_dir=ROOT_DIR / "indexes" / "dense",
        top_k=top_k,
    )

    hybrid = Hybrid(
        retrievers=[bm25, dense],
        top_k=top_k,
        rrf_k=60,
    )

    return [bm25, dense, hybrid]


def build_agents(retriever, rewriter: QueryRewriter, top_k: int):
    standard = StandardAgent(retriever=retriever, top_k=top_k)
    fusion = FusionAgent(retriever=retriever, rewriter=rewriter, top_k=top_k, rrf_k=60)
    return [standard, fusion]


def build_chunk_judge_chain():
    llm = ChatOpenAI(model=MODEL, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=ChunkJudge)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Você é um avaliador de relevância para benchmark de recuperação em documento oficial (BNCC). "
             "Você não inventa fatos. Você avalia apenas se o trecho ajuda a responder a consulta."),
            ("human",
             "Consulta:\n{query}\n\n"
             "Trecho:\nchunk_id: {chunk_id}\ntext: {chunk_text}\n\n"
             "Atribua score 0-3:\n"
             "3 = responde diretamente e contém evidência clara\n"
             "2 = muito relacionado e ajuda bastante\n"
             "1 = relacionado indiretamente / contexto fraco\n"
             "0 = irrelevante ou genérico\n\n"
             "Regra extra:\n"
             "- Se a consulta contém um código BNCC (ex: EM13MAT303), só dê 3 se o trecho contém esse código "
             "ou descreve exatamente essa habilidade.\n\n"
             "{format_instructions}")
        ]
    )

    chain = prompt | llm | parser
    return chain, parser


def judge_one(chain, parser, query: str, chunk_id: str, chunk_text: str) -> ChunkJudge:
    chunk_text = preview(chunk_text, max_chars=900)
    return chain.invoke(
        {
            "query": query,
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def main():
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        bench = json.load(f)

    chunk_text_map = load_chunk_text_map(CHUNKS_PATH)

    retrievers = build_retrievers(top_k=TOP_K_PER_SYSTEM)
    rewriter = QueryRewriter(model=MODEL, n=3)

    chain, parser = build_chunk_judge_chain()

    updated = []
    for item in bench:
        qid = item.get("id", "")
        query = item.get("query", "")

        print("\n" + "=" * 90)
        print(f"[JUDGE] id={qid}")
        print(query)

        existing_relevant: dict[str, int] = item.get("relevant") or {}
        existing_relevant = {str(k): int(v) for k, v in existing_relevant.items()}

        scored: dict[str, int] = dict(existing_relevant)

        seen: set[str] = set()

        for retriever in retrievers:
            agents = build_agents(retriever, rewriter, top_k=TOP_K_PER_SYSTEM)

            for agent in agents:
                results = agent.retrieve(query)[:TOP_K_PER_SYSTEM]

                for r in results:
                    cid = r.node.node_id

                    if cid in seen:
                        continue
                    seen.add(cid)

                    if cid in existing_relevant:
                        continue

                    text = chunk_text_map.get(cid, "")
                    if not text.strip():
                        scored[cid] = 0
                        continue

                    judged = judge_one(chain, parser, query=query, chunk_id=cid, chunk_text=text)
                    scored[cid] = int(judged.score)

                    print(f"  chunk={cid} | score={judged.score}")

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:TOP_SAVE]

        item["relevant"] = {cid: int(s) for cid, s in top}

        updated.append(item)

        print("Saved relevant (top):", item["relevant"])

    out_path = QUERIES_PATH.parent / "queries_judged.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
