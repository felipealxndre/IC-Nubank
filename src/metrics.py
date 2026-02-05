import math
from pathlib import Path
import json




# recall = identificar itens 
def recall(retrieved_chunks: list[str], benchmark: dict[str, int], k: int) -> float:
    rel_ids = [cid for cid, imp in benchmark.items()]
    if len(rel_ids) == 0:
        return 0.0
    retrieved_chunks = retrieved_chunks[:k]
    hits = sum(1 for cid in retrieved_chunks if cid in rel_ids)
    return hits / len(rel_ids)

def mean_reciprocal_rank(ranked_ids, relevant, k):
    rel_ids = {cid for cid, imp in relevant.items() if int(imp) > 0}
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in rel_ids:
            return 1.0 / rank
    return 0.0

def normalized_discounted_cumulative_gain(retrieved_chunks: list[str], benchmark: dict[str, int], k: int) -> float:
    """
    nDCG@k para relevância graduada (0,1,2,3...).

    DCG = sum_{i=1..k} ( (2^rel_i - 1) / log2(i+1) )
    nDCG = DCG / IDCG

    - Se você usa relevance 0/1, ainda funciona (vira DCG binário).
    - Com 0/1/2/3 fica bem mais informativo para paper.
    """

    def dcg(ids: list[str]) -> float:
        s = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            rel = int(benchmark.get(cid, 0))
            gain = (2**rel - 1)
            s += gain / math.log2(i + 1)
        return s

    dcg_val = dcg(retrieved_chunks)

    # ideal ranking: ordenar os relevantes por maior relevância primeiro
    ideal_sorted = sorted(benchmark.items(), key=lambda x: int(x[1]), reverse=True)
    ideal_ids = [cid for cid, _ in ideal_sorted]

    idcg_val = dcg(ideal_ids)

    # se idcg = 0, significa que não tem relevante (ou tudo 0)
    if idcg_val == 0.0:
        return 0.0

    return dcg_val / idcg_val

