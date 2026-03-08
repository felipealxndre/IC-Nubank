# IC-Nubank: Benchmark de recuperação sobre a BNCC

Projeto de avaliação de sistemas de recuperação de informação (RAG) sobre o documento da Base Nacional Comum Curricular (BNCC). Compara retrievers léxicos (BM25), vetoriais (Dense) e híbridos, com agentes padrão e RAG-Fusion, usando métricas Recall, MRR e nDCG.

---

## Requisitos

- Python 3.10+
- Chave da API OpenAI (embeddings e LLM para reescrita de query e judge)

Configure a chave em um arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sua_chave_aqui
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Se usar stopwords do NLTK, baixe o recurso uma vez:

```python
import nltk
nltk.download("stopwords")
```

---

## Estrutura do projeto

```
IC-Nubank/
  bench/                 # Benchmark de avaliação
    queries.json        # Queries e gold (chunk_id -> nota 0-3)
    queries_judged.json # Gold julgado por LLM (com texto e rationale)
    queries_with_text.json  # Queries com trechos de texto por chunk (opcional)
  data/
    raw/                # PDF da BNCC (fonte)
    processed/
      chunks.jsonl      # Chunks com text_raw, text_lex, metadata
    results/            # Saída da avaliação (CSV, gráficos, tabela)
  indexes/              # Índices persistidos (BM25, FAISS/dense)
  src/
    main.py             # Pipeline de avaliação (retrieval + métricas)
    judge.py            # LLM-as-judge: gera/atualiza gold em queries_judged.json
    agents.py           # StandardAgent e FusionAgent (RAG-Fusion)
    query_rewrite.py    # Reescrita de query para Fusion
    retrievers/         # BM25, Dense (OpenAI + FAISS), Hybrid (RRF)
    nodes_from_chunks.py
    build_corpus.py     # PDF -> chunks.jsonl (text_raw, text_lex)
    metrics.py          # Recall@k, MRR@k, nDCG@k
    utils/reporting.py  # Geração de tabelas e gráficos
```

---

## Fluxo de uso

### 1. Preparar o corpus (uma vez)

Coloque o PDF da BNCC em `data/raw/` (ex.: `bncc.pdf`) e rode:

```bash
cd src && python build_corpus.py
```

Isso gera `data/processed/chunks.jsonl` com campos `chunk_id`, `text_raw`, `text_lex` e metadados. Os índices (BM25 e Dense) são criados na primeira execução do `main.py` ou do `judge.py` e persistidos em `indexes/`.

### 2. Avaliar os sistemas de retrieval

O script principal lê o benchmark em `bench/queries_judged.json`, roda cada query nos retrievers (Dense, BM25, Hybrid) e nos agentes (Standard, Fusion), e calcula Recall@k, MRR@k e nDCG@k:

```bash
cd src && python main.py
```

Resultados são gravados em `data/results/`:

- `per_query.csv` – métricas por (query, agent, retriever)
- `summary.csv` e `table_summary.md` – médias por sistema
- `table_summary.png` – tabela em imagem
- `plot_ndcg.png` e `plot_mrr.png` – gráficos de barras

### 3. Gerar ou atualizar o gold (LLM-as-judge)

O judge usa os mesmos retrievers e agentes para obter candidatos por query, avalia cada par (query, trecho) com um LLM (rubrica 0–3 e regra para códigos BNCC) e salva o gold em `bench/queries_judged.json` (formato enriquecido: chunk_id, text, nota, rationale):

```bash
cd src && python judge.py
```

Requer `OPENAI_API_KEY`. O arquivo `queries.json` em `bench/` é a entrada de queries; o judge lê os chunks em `data/processed/chunks.jsonl`.

---

## Retrievers e agentes

- **BM25**: retriever léxico (stemming em português), índice em `indexes/bm25`. Usa o campo `text_lex` dos chunks.
- **Dense**: embeddings OpenAI (text-embedding-3-small) + FAISS, índice em `indexes/dense`. Usa `text_raw`.
- **Hybrid**: fusão por RRF (Reciprocal Rank Fusion) dos rankings do BM25 e do Dense, sem peso extra por retriever.

Agentes:

- **StandardAgent**: uma query, um ranking por retriever.
- **FusionAgent**: reescrita da query (LangChain + OpenAI), várias queries; fusão dos rankings por RRF.

---

## Métricas

- **Recall@k**: fração dos documentos relevantes (nota > 0 no gold) que aparecem no top-k recuperado.
- **MRR@k**: inverso do rank do primeiro documento relevante no top-k (0 se não houver relevante).
- **nDCG@k**: ganho acumulado descontado normalizado, com relevância graduada 0–3.

O gold pode ser um dicionário `{ chunk_id: nota }` ou uma lista de `{ "chunk_id", "text", "nota", "rationale" }`; o `main.py` normaliza para dict internamente.

---

## Observações

- Os caminhos assumem execução a partir da pasta `src` ou com a raiz do repositório como diretório de trabalho; `root_dir` é obtido por `Path(__file__).resolve().parents[1]`.
- Se a fonte Segoe UI não estiver instalada, os gráficos em `utils/reporting.py` podem emitir avisos de fonte; os plots ainda são gerados com fallback.
