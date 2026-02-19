import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from train_lemma_lstm import CharVocab, Seq2SeqLSTM
from utils import is_word, project_root


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


def iter_lemma_tsv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            word, lemma = parts[0], parts[1]
            yield word, lemma


def lev_distance(a: str, b: str) -> int:
    # классический DP, оптимизирован по памяти
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


@torch.no_grad()
def batch_lemmatize(words: List[str], vocab: CharVocab, model: Seq2SeqLSTM, device: torch.device) -> List[str]:
    xs = [vocab.encode(w) for w in words]
    lens = [len(x) for x in xs]
    max_x = max(lens) if lens else 1

    x_pad = []
    for x in xs:
        x_pad.append(x + [vocab.pad_id] * (max_x - len(x)))

    x = torch.tensor(x_pad, dtype=torch.long, device=device)
    x_len = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = min(64, max_x + 8)
    pred_ids = model.greedy(x, x_len, vocab=vocab, max_len=max_len)
    return [vocab.decode(seq) for seq in pred_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--report-dir", default="")
    ap.add_argument("--print-every", type=int, default=50000)
    args = ap.parse_args()

    root = project_root()
    lemma_dir = Path(args.model_dir) if args.model_dir else (root / "models" / "lemma_lstm")
    data_path = Path(args.data) if args.data else (root / "data" / "processed" / "lemma_test.tsv")
    report_dir = Path(args.report_dir) if args.report_dir else (root / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, model = load_lemma_model(lemma_dir, device)

    total = 0
    correct = 0
    skipped = 0
    dist_sum = 0
    char_sum = 0

    buf_words: List[str] = []
    buf_gold: List[str] = []

    def flush():
        nonlocal total, correct, skipped, dist_sum, char_sum, buf_words, buf_gold
        if not buf_words:
            return
        preds = batch_lemmatize(buf_words, vocab, model, device)
        for w, g, p in zip(buf_words, buf_gold, preds):
            total += 1
            if p == g:
                correct += 1
            dist_sum += lev_distance(p, g)
            char_sum += max(1, len(g))
        buf_words, buf_gold = [], []

    for i, (w, g) in enumerate(iter_lemma_tsv(data_path), 1):
        # если вдруг в тесте попадётся не-слово, просто пропустим
        if not is_word(w):
            skipped += 1
            continue
        buf_words.append(w)
        buf_gold.append(g)
        if len(buf_words) >= args.batch:
            flush()
        if args.print_every and (i % args.print_every == 0):
            flush()
            acc = correct / total if total else 0.0
            cer = dist_sum / char_sum if char_sum else 0.0
            print(f"[lemma] seen={i} scored={total} acc_exact={acc:.4f} cer={cer:.4f} skipped={skipped}")

    flush()

    acc = correct / total if total else 0.0
    cer = dist_sum / char_sum if char_sum else 0.0

    metrics = {
        "data": str(data_path),
        "model_dir": str(lemma_dir),
        "total_words_scored": total,
        "skipped_non_words": skipped,
        "exact_match_accuracy": acc,
        "cer": cer,
        "avg_edit_distance": (dist_sum / total) if total else 0.0,
    }

    out_path = report_dir / "lemma_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[lemma] exact={acc:.4f} cer={cer:.4f} avg_ed={metrics['avg_edit_distance']:.4f}")
    print(f"[lemma] saved: {out_path}")


if __name__ == "__main__":
    main()
