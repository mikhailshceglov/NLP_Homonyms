import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import project_root


def _get(obj, keys):
    for k in keys:
        if k in obj:
            return obj[k]
    return None


def iter_pos_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            tokens = _get(obj, ["tokens", "words", "x"])
            tags = _get(obj, ["tags", "labels", "y", "pos"])
            if tokens is None or tags is None:
                raise RuntimeError(f"Bad jsonl row, expected tokens/tags keys, got: {list(obj.keys())}")
            yield tokens, tags


@torch.no_grad()
def predict_tags(tokens, tokenizer, model, device, max_length: int):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    wids = enc.word_ids()

    out = [None] * len(tokens)
    for i, wid in enumerate(wids):
        if wid is None:
            continue
        if wid < len(out) and out[wid] is None:
            out[wid] = model.config.id2label[int(pred_ids[i])]
    return out  # может содержать None у токенов, которые не попали из-за truncation


def f1_from_counts(tp, pred_cnt, gold_cnt):
    labels = sorted(gold_cnt.keys())
    per = {}
    macro_sum = 0.0
    weighted_sum = 0.0
    total_support = 0

    for l in labels:
        tp_l = tp.get(l, 0)
        p_den = pred_cnt.get(l, 0)
        r_den = gold_cnt.get(l, 0)

        prec = tp_l / p_den if p_den else 0.0
        rec = tp_l / r_den if r_den else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        support = r_den
        per[l] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

        macro_sum += f1
        weighted_sum += f1 * support
        total_support += support

    macro_f1 = macro_sum / len(labels) if labels else 0.0
    weighted_f1 = weighted_sum / total_support if total_support else 0.0
    return macro_f1, weighted_f1, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--report-dir", default="")
    ap.add_argument("--top-confusions", type=int, default=20)
    ap.add_argument("--print-every", type=int, default=20000)
    args = ap.parse_args()

    root = project_root()
    model_dir = Path(args.model_dir) if args.model_dir else (root / "models" / "pos_transformer")
    data_path = Path(args.data) if args.data else (root / "data" / "processed" / "pos_test.jsonl")
    report_dir = Path(args.report_dir) if args.report_dir else (root / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()

    total = 0
    correct = 0
    skipped = 0
    mismatched_len = 0

    gold_cnt = Counter()
    pred_cnt = Counter()
    tp = Counter()
    conf = Counter()  # (gold, pred) -> count

    for idx, (tokens, gold_tags) in enumerate(iter_pos_jsonl(data_path), 1):
        if len(tokens) != len(gold_tags):
            m = min(len(tokens), len(gold_tags))
            tokens = tokens[:m]
            gold_tags = gold_tags[:m]
            mismatched_len += 1

        pred_tags = predict_tags(tokens, tokenizer, model, device, args.max_length)

        for g, p in zip(gold_tags, pred_tags):
            if p is None:
                skipped += 1
                continue
            total += 1
            gold_cnt[g] += 1
            pred_cnt[p] += 1
            conf[(g, p)] += 1
            if g == p:
                correct += 1
                tp[g] += 1

        if args.print_every and (idx % args.print_every == 0):
            acc = correct / total if total else 0.0
            print(f"[pos] seen={idx} tokens={total} acc={acc:.4f} skipped={skipped}")

    acc = correct / total if total else 0.0
    macro_f1, weighted_f1, per_label = f1_from_counts(tp, pred_cnt, gold_cnt)

    # топ ошибок (кроме правильных)
    wrong = [((g, p), c) for (g, p), c in conf.items() if g != p]
    wrong.sort(key=lambda x: x[1], reverse=True)
    top_wrong = wrong[: args.top_confusions]

    metrics = {
        "data": str(data_path),
        "model_dir": str(model_dir),
        "max_length": args.max_length,
        "total_tokens_scored": total,
        "skipped_tokens_truncated": skipped,
        "mismatched_len_rows": mismatched_len,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": per_label,
        "top_confusions": [
            {"gold": g, "pred": p, "count": c} for ((g, p), c) in top_wrong
        ],
    }

    out_path = report_dir / "pos_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pos] accuracy={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")
    if top_wrong:
        print("[pos] top confusions:")
        for (g, p), c in top_wrong:
            print(f"  {g} -> {p}: {c}")
    print(f"[pos] saved: {out_path}")


if __name__ == "__main__":
    main()
