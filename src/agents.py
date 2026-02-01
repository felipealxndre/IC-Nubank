from collections import defaultdict
from query_rewrite import QueryRewriter


class StandardAgent:
    """
    RAG padrão: uma única query(a do user)
    """
    def __init__(self, retriever, top_k: int):
        self.retriever = retriever
        self.top_k = top_k
    
    def retrieve(self, query: str):
        return self.retriever.retrieve(query)[:self.top_k]
    

class FusionAgent:
    """
    RAG-Fusion: gera variações de query, roda retrieval por query gerada e funde com RRF.
    """
    def __init__(self, retriever, rewriter: QueryRewriter = QueryRewriter(), top_k: int = 5, rrf_k: int = 10):
        self.retriever = retriever
        self.rewriter = rewriter
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str):
        # generating queries from original query
        queries = self.rewriter.rewrite(query)
        # adiciona a query original
        queries = [query] + queries

        rankings = [self.retriever.retrieve(q) for q in queries]

        scores = defaultdict(float)
        by_id = {}
        for rank_list in rankings:
            for rank, item in enumerate(rank_list, start=1):
                node_id = item.node.node_id
                
                scores[node_id] += 1.0/(self.rrf_k + rank)
                by_id[node_id] = item
        
        fused = []
        for node_id, s in scores.items():
            base = by_id[node_id]
            base.score = float(s)
            fused.append(base)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[: self.top_k]
