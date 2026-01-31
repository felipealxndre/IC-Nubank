
import re
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import json

# faz com que os ids fiquem com nomes seguros
def make_safe_id(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

# função pra salvar os jsonl, tanto chunks quanto pages
def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    
    # definindo os paths
    input_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir = Path(__file__).parent.parent / "data" / "processed"


    # load bnc pdf -> Documents
    loader = SimpleDirectoryReader(
        input_dir = input_dir,
        required_exts = [".pdf"]
    )
    documents = loader.load_data()

    # saving raw raw docs
    pages = []
    for i, d in enumerate(documents):
        pages.append(
            {
                "doc_id": i,
                "text": (d.get_content() or "").strip(),
                "metadata": dict(d.metadata or {}),
            }
        )

    # salvando no jsonl as pages
    write_jsonl(path=output_dir / "pages.jsonl", rows=pages)

    # splitando, agora ao invés de páginas vão ser em chunks
    # criando o splitter
    splitter = SentenceSplitter(
        # esse chunk_size precisa ser testado se está adequado
        chunk_size = 512,
        # colocando um overlap para não perder contexto entre uma quebra e outra
        chunk_overlap = 50,
    )

    # nodes são partes identificaveis dos chunks
    nodes = splitter.get_nodes_from_documents(documents=documents)

    chunks = []
    for i, n in enumerate(nodes):
        meta = dict(n.metadata or {})
        file_name = meta.get("file_name") or "unknown_file"
        page = meta.get("page_label") or "na"
        # vamos colocar um id rastreavel
        chunk_id = f"{make_safe_id(file_name)}__p{make_safe_id(str(page))}__c{i:05d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": (n.get_content() or "").strip(),
                "metadata": meta,
            }
        )
    write_jsonl(output_dir / "chunks.jsonl", rows=chunks)


