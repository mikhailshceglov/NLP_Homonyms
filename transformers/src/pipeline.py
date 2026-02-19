from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train_lemma_lstm import CharVocab, Seq2SeqLSTM
from utils import is_punct, is_word, project_root, split_sentences, tokenize_sentence


def load_vocab(vocab_path: Path) -> CharVocab:
    itos = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return CharVocab(stoi=stoi, itos=itos)


def load_pos_model(pos_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(str(pos_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(pos_dir)).to(device).eval()
    return tokenizer, model


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
def pos_tag(tokens: List[str], tokenizer, model, device: torch.device) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    wids = enc.word_ids()

    out = [None] * len(tokens)
    for i, wid in enumerate(wids):
        if wid is None:
            continue
        if out[wid] is None:
            out[wid] = model.config.id2label[int(pred_ids[i])]
    return [t if t is not None else "X" for t in out]


@torch.no_grad()
def lemmatize(tokens: List[str], vocab: CharVocab, model: Seq2SeqLSTM, device: torch.device) -> List[str]:
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
    ap.add_argument("--pos-dir", default="")
    ap.add_argument("--lemma-dir", default="")
    args = ap.parse_args()

    root = project_root()
    pos_dir = Path(args.pos_dir) if args.pos_dir else (root / "models" / "pos_transformer")
    lemma_dir = Path(args.lemma_dir) if args.lemma_dir else (root / "models" / "lemma_lstm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_tokenizer, pos_model = load_pos_model(pos_dir, device)
    vocab, lemma_model = load_lemma_model(lemma_dir, device)

    text = Path(args.in_file).read_text(encoding="utf-8") if args.in_file else args.text
    sentences = split_sentences(text) if "\n" in text else ([text.strip()] if text.strip() else [])

    for line in sentences:
        tokens = tokenize_sentence(line)  # токены + пунктуация (для контекста POS)
        tags = pos_tag(tokens, pos_tokenizer, pos_model, device)
        lemmas = lemmatize(tokens, vocab, lemma_model, device)

        # ВЫВОД по ТЗ: без пунктуации, разделитель пробел, одна строка на предложение
        out = []
        for w, l, t in zip(tokens, lemmas, tags):
            if is_punct(w):
                continue
            out.append(f"{w}{{{l}={t}}}")
        print(" ".join(out))


if __name__ == "__main__":
    main()
