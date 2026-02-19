from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import numpy as np
except Exception:
    np = None


PUNCT = {",", ".", "?", "!"}


def project_root() -> Path:
    # .../transformers/src/utils.py -> .../transformers
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9]+", tok))


def is_punct(tok: str) -> bool:
    return tok in PUNCT


def split_sentences(text: str) -> List[str]:
    # Каждая строка входа = одно предложение
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def tokenize_sentence(line: str) -> List[str]:
    """
    Вход по заданию уже "токен1 токен2 ... токенN".
    Но на всякий случай:
    - если пунктуация прилеплена к слову (например "пирожки,"),
      отделяем хвостовые , . ? !
    - если токен состоит только из пунктуации (например "?!"),
      разбиваем на отдельные символы
    """
    raw = line.strip().split()
    out: List[str] = []

    for tok in raw:
        if not tok:
            continue

        if all(ch in PUNCT for ch in tok):
            out.extend(list(tok))
            continue

        # отделяем хвостовую пунктуацию
        suffix: List[str] = []
        while len(tok) > 1 and tok[-1] in PUNCT:
            suffix.append(tok[-1])
            tok = tok[:-1]

        if tok:
            out.append(tok)
        for p in reversed(suffix):
            out.append(p)

    return out
