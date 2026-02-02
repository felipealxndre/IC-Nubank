import json
import re
import unicodedata
from pathlib import Path

import nltk
from nltk.corpus import stopwords


def _normalize_token(s: str) -> str:
    """
    Normaliza 1 token para ficar compatível com text_lex:
    - lowercase
    - remove acentos
    - remove caracteres especiais (mantém a-z, 0-9 e underscore)
    - normaliza espaços
    """
    if not s:
        return ""

    s = s.strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = re.sub(r"[^a-z0-9_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


if __name__ == "__main__":
    # garante que o corpus existe
    nltk.download("stopwords", quiet=True)

    root_dir = Path(__file__).resolve().parents[1]
    out_path = root_dir / "data" / "stopwords.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_sw = set(stopwords.words("portuguese"))

    # stopwords customizadas
    custom_sw = {
        "bncc", "base", "nacional", "comum", "curricular",
        "competência", "competências", "habilidade", "habilidades",
        "aprendizagem", "aprendizagens",
        "estudante", "estudantes",
        "ensino", "fundamental", "médio",
    }

    # normaliza tudo
    all_sw = set()
    for w in base_sw:
        ww = _normalize_token(w)
        if ww:
            all_sw.add(ww)

    for w in custom_sw:
        ww = _normalize_token(w)
        if ww:
            all_sw.add(ww)

    all_sw = {w for w in all_sw if w and w != "_"}

    payload = {
        "language": "portuguese",
        "count": len(all_sw),
        "stopwords": sorted(all_sw),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
