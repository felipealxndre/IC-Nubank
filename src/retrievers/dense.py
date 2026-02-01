from pathlib import Path
import faiss
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode


class DenseRetriever:
    """
    Retriever denso (embeddings OpenAI) + índice FAISS com persistência.
    Interface: retrieve(query) -> list[NodeWithScore]
    """

    def __init__(
        self,
        nodes: list[TextNode],
        persist_dir: Path,
        top_k: int,
        embedding_model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        self.top_k = top_k
        persist_dir.mkdir(parents=True, exist_ok=True)

        # embeddings OpenAI
        embed_model = OpenAIEmbedding(model=embedding_model, dimensions=dimensions)
        Settings.embed_model = embed_model

        # carregando ou criando o índice FAISS
        has_index = (persist_dir / "docstore.json").exists() and (persist_dir / "index_store.json").exists()
        if has_index:
            vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(persist_dir),
            )
            self._index = load_index_from_storage(storage_context)
        else:
            dim = len(embed_model.get_text_embedding("teste"))
            faiss_index = faiss.IndexFlatL2(dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
            self._index.storage_context.persist(persist_dir=str(persist_dir))

        self._retriever = self._index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, query: str):
        return self._retriever.retrieve(query)[: self.top_k]