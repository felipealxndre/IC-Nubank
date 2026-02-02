
import re
from pathlib import Path
import unicodedata
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import json
from nltk.corpus import stopwords

# definindo os paths
data_dir = Path(__file__).parent.parent / "data"

# preparando o conjunto de stopwords
with data_dir.joinpath("stopwords.json").open("r", encoding="utf-8") as f:
    sw_payload = json.load(f)
stop_set = set(sw_payload.get("stopwords", []))


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

# limpando o texto para abordagem lexical
def clean_text(text: str) -> str:
    # desfaz hifenização no fim de linha
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # quebra de linha vira espaço
    text = text.replace("\n", " ")
    # lowercase
    text = text.lower()
    # remove acentos
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # remove caracteres especiais (mantém a-z, 0-9 e espaço)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    # normaliza espaços
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_set]
    return " ".join(tokens)

if __name__ == "__main__":
    
    

    # load bnc pdf -> Documents
    loader = SimpleDirectoryReader(
        input_dir = data_dir / "raw",
        required_exts = [".pdf"]
    )
    documents = loader.load_data()

    # saving raw raw docs - (raw and lexical version)
    pages = []
    for i, d in enumerate(documents):
        raw = (d.get_content() or "").strip()
        lex = clean_text(raw)
        lex = remove_stopwords(lex)

        pages.append(
            {
                "doc_id": i,
                "text_raw": raw,
                "text_lex": lex,
                "metadata": dict(d.metadata or {}),
            }
        )

    # salvando no jsonl as pages
    write_jsonl(path=data_dir / "processed" / "pages.jsonl", rows=pages)

    # splitando, agora ao invés de páginas vão ser em chunks
    # criando o splitter
    splitter = SentenceSplitter(
        # esse chunk_size precisa ser testado se está adequado
        chunk_size = 512,
        # colocando um overlap para não perder contexto entre uma quebra e outra
        chunk_overlap = 100,
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

        raw = (n.get_content() or "").strip()
        lex = clean_text(raw)
        lex = remove_stopwords(lex)
    
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text_raw": raw,
                "text_lex": lex,
                "metadata": meta,
            }
        )
    write_jsonl(data_dir / "processed" / "chunks.jsonl", rows=chunks)


