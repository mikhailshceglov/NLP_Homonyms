from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from utils import project_root, set_seed


class PosJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, tag2id: Dict[str, int], max_length: int = 256):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        item = self.rows[idx]
        tokens: List[str] = item["tokens"]
        tags: List[str] = item["tags"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = enc.word_ids()
        labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_wid:
                labels.append(self.tag2id.get(tags[wid], 0))
            else:
                labels.append(-100)
            prev_wid = wid

        enc["labels"] = labels
        return enc


def compute_metrics_builder(id2tag: Dict[int, str]):
    from seqeval.metrics import accuracy_score, f1_score

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids

        true_tags = []
        pred_tags = []
        for sent_pred, sent_lab in zip(preds, labels):
            t_true = []
            t_pred = []
            for pr, lb in zip(sent_pred, sent_lab):
                if lb == -100:
                    continue
                t_true.append(id2tag[int(lb)])
                t_pred.append(id2tag[int(pr)])
            true_tags.append(t_true)
            pred_tags.append(t_pred)

        return {"f1": f1_score(true_tags, pred_tags), "acc": accuracy_score(true_tags, pred_tags)}

    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai-forever/ruBert-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    root = project_root()
    data_dir = root / "data" / "processed"
    out_dir = root / "models" / "pos_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    tagset = json.loads((data_dir / "pos_tagset.json").read_text(encoding="utf-8"))
    tag2id = {t: i for i, t in enumerate(tagset)}
    id2tag = {i: t for t, i in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(tagset),
        id2label=id2tag,
        label2id=tag2id,
    )

    train_ds = PosJsonlDataset(data_dir / "pos_train.jsonl", tokenizer, tag2id, max_length=args.max_length)
    dev_ds = PosJsonlDataset(data_dir / "pos_dev.jsonl", tokenizer, tag2id, max_length=args.max_length)

    collator = DataCollatorForTokenClassification(tokenizer)
    use_fp16 = bool(args.fp16) and torch.cuda.is_available()

    targs = TrainingArguments(
        output_dir=str(out_dir / "_run"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=200,
        fp16=use_fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_builder(id2tag),
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
