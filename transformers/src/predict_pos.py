from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import is_punct, project_root, split_sentences, tokenize_sentence


def tag_tokens(tokens: List[str], tokenizer, model, device: torch.device) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--in-file", default="")
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()

    root = project_root()
    model_dir = Path(args.model_dir) if args.model_dir else (root / "models" / "pos_transformer")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    text = Path(args.in_file).read_text(encoding="utf-8") if args.in_file else args.text
    sentences = split_sentences(text) if "\n" in text else [text.strip()] if text.strip() else []

    for line in sentences:
        tokens = tokenize_sentence(line)
        tags = tag_tokens(tokens, tokenizer, model, device)

        # по заданию пунктуацию в вывод не включаем
        out = []
        for w, t in zip(tokens, tags):
            if is_punct(w):
                continue
            # predict_pos не знает лемму, поэтому лемма=сам токен (для отладки)
            out.append(f"{w}{{{w}={t}}}")

        print(" ".join(out))


if __name__ == "__main__":
    main()
