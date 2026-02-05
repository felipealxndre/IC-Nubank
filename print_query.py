import json
import sys
from rich import print

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

queries_path = "bench/queries.json"
chunks_path = "data/processed/chunks.jsonl"

with open(queries_path, 'r') as file:
    itens = json.load(file)

chunks = {}

with open(chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)                 # agora é dict
        chunk_id = obj["chunk_id"]
        chunks[chunk_id] = obj

for item in itens:
    query_id = item['id']
    if query_id == "q5":
        query = item['query']
        relevants = item['relevant']
        print(f"\n[yellow]Query: {query} [/yellow]\n")
        for id, importance in relevants.items():
            # pegando o chunk
            chunk = chunks[id]
            print(f"[green]Chunk ID: {id}[/green], importance: {importance}\n")
            print(f"\n{chunk['text_raw']}\n")