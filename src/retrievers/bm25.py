from pathlib import Path
import re
from llama_index.core.schema import TextNode
import Stemmer
from llama_index.retrievers.bm25 import BM25Retriever as bm25_retriever
from build_corpus import clean_text

class BM25Retriever:
    def __init__(
        self,
        nodes: list[TextNode],
        persist_dir: Path,
        top_k: int,
        top_n: int = 50,
        code_bonus: float = 5.0,
    ):
        
        self.top_k = top_k
        self.top_n = top_n
        self.code_bonus = float(code_bonus)

        persist_dir.mkdir(parents=True, exist_ok=True)

        # Cria um novo índice BM25
        stemmer = Stemmer.Stemmer("portuguese")

        self._retriever = bm25_retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_n,
            stemmer=stemmer,
            language="portuguese",
        )
        # salva o índice para persistência
        self._retriever.persist(str(persist_dir))


    def _bncc_codes_identifier(self, text: str) -> list[str]:
        pattern = r"\b(?:em\d{2}[a-z]{2,4}\d{2,4}|ef\d{2}[a-z]{1,4}\d{1,4})\b"
        found = re.findall(pattern, text)
        seen = set()
        codes = []
        for code in found:
            if code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    def retrieve(self, query: str):
        query = clean_text(query)
        
        print(f"[BM25Retriever] Query after cleaning: \n\n{query}\n\n")

        candidates = self._retriever.retrieve(query)[: self.top_n]    
        codes = self._bncc_codes_identifier(query)
        # se tiver códigos BNCC na query, dá um boost nos resgates que os contenham
        if codes:
            for item in candidates:
                doc_text = (item.node.text or "")
                hits = sum(1 for c in codes if c in doc_text)

                # bônus proporcional ao número de códigos encontrados (match exato)
                if hits > 0:
                    base = float(item.score or 0.0)
                    item.score = base + (hits * self.code_bonus)

            # ordena após boost
            candidates.sort(key=lambda x: float(x.score or 0.0), reverse=True)

        return candidates[: self.top_k]   

    