from collections import defaultdict

class Hybrid:
    """
    Retriever híbrido via Rank Fusion (RRF).

    - Recebe uma lista de retrievers (ex.: [bm25, dense])
    - Para uma query:
        1) roda retrieve em cada retriever
        2) aplica RRF para fundir rankings
        3) retorna Top-K

    RRF score:
        score(doc) = sum_{retriever} 1 / (rrf_k + rank(doc))
    """

    def __init__(self, retrievers: list, top_k: int = 5, rrf_k: int = 10):
        self.retrievers = retrievers
        self.top_k = top_k
        self.rrf_k = rrf_k
    
    def retrieve(self, query: str):
        rankings = []
        for r in self.retrievers:
            results = r.retrieve(query)[: self.top_k]
            rankings.append(results)
        scores = defaultdict(float)
        by_id = {}

        for rank_list in rankings:
            for rank, item in enumerate(rank_list, start = 1):
                node_id = item.node.node_id
                scores[node_id] += 1.0/(self.rrf_k + rank)
                by_id[node_id] = item
        
        fused = []
        for node_id, s in scores.items():
            base = by_id[node_id]

            # guarda o score RRF
            base.node.metadata = base.node.metadata or {}
            base.node.metadata["rrf_score"] = float(s)

            # score final do item vira o score do RRF
            base.score = float(s)
            fused.append(base)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[: self.top_k]