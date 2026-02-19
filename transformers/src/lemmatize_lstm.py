from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from train_lemma_lstm import CharVocab, Seq2SeqLSTM
from utils import is_punct, is_word, project_root, split_sentences, tokenize_sentence


def load_vocab(vocab_path: Path) -> CharVocab:
    itos = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return CharVocab(stoi=stoi, itos=itos)


def load_lemma_model(lemma_dir: Path, device: torch.device):
    vocab = load_vocab(lemma_dir / "vocab.json")
    state = torch.load(lemma_dir / "model.pt", map_location="cpu")

    vocab_size, emb_dim = state["emb.weight"].shape
    hid_dim = state["proj.weight"].shape[1]

    if len(vocab.itos) != vocab_size:
        raise RuntimeError(
            f"vocab.json size={len(vocab.itos)} != checkpoint vocab={vocab_size}. "
            "Нужно, чтобы vocab.json соответствовал model.pt."
        )

    model = Seq2SeqLSTM(vocab_size=vocab_size, emb=emb_dim, hid=hid_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return vocab, model


@torch.no_grad()
def lemmatize_tokens(tokens: List[str], vocab: CharVocab, model: Seq2SeqLSTM, device: torch.device) -> List[str]:
    out = []
    for t in tokens:
        if is_punct(t) or (not is_word(t)):
            out.append(t)
            continue
        x_ids = vocab.encode(t)
        x = torch.tensor([x_ids], dtype=torch.long, device=device)
        x_len = torch.tensor([len(x_ids)], dtype=torch.long, device=device)
        pred = model.greedy(x, x_len, vocab=vocab, max_len=max(8, len(x_ids) + 4))[0]
        out.append(vocab.decode(pred))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--in-file", default="")
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()

    root = project_root()
    lemma_dir = Path(args.model_dir) if args.model_dir else (root / "models" / "lemma_lstm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, model = load_lemma_model(lemma_dir, device)

    text = Path(args.in_file).read_text(encoding="utf-8") if args.in_file else args.text
    sentences = split_sentences(text) if "\n" in text else ([text.strip()] if text.strip() else [])

    for line in sentences:
        tokens = tokenize_sentence(line)
        lemmas = lemmatize_tokens(tokens, vocab, model, device)

        # По ТЗ: без пунктуации, токены через пробел.
        # Здесь нет POS, поэтому ставим тег X (для отладки).
        out = []
        for w, l in zip(tokens, lemmas):
            if is_punct(w):
                continue
            out.append(f"{w}{{{l}=X}}")
        print(" ".join(out))


if __name__ == "__main__":
    main()
