from pathlib import Path
from llama_index.core.schema import TextNode
import Stemmer
from llama_index.retrievers.bm25 import BM25Retriever as bm25_retriever
from build_corpus import clean_text, remove_stopwords

class BM25Retriever:
    def __init__(self, nodes: list[TextNode], persist_dir: Path, top_k: int):
        self.top_k = top_k
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Cria um novo índice BM25
        stemmer = Stemmer.Stemmer("portuguese")

        self._retriever = bm25_retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
            stemmer=stemmer,
            language="portuguese",
        )
        # salva o índice para persistência
        self._retriever.persist(str(persist_dir))

    def retrieve(self, query: str):
        query = clean_text(query)
        query = remove_stopwords(query)
        print(f"[BM25Retriever] Query after cleaning: \n\n{query}\n\n")
        return self._retriever.retrieve(query)[: self.top_k]        

    