from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import project_root, set_seed


SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, pairs: List[Tuple[str, str]], min_freq: int = 1):
        freq: Dict[str, int] = {}
        for w, l in pairs:
            for ch in w:
                freq[ch] = freq.get(ch, 0) + 1
            for ch in l:
                freq[ch] = freq.get(ch, 0) + 1

        itos = SPECIALS[:]
        for ch, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and ch not in itos:
                itos.append(ch)
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(ch, unk) for ch in s]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            ch = self.itos[i]
            if ch in ("<pad>", "<sos>", "<eos>"):
                continue
            out.append(ch)
        return "".join(out)

    @property
    def pad_id(self): return self.stoi["<pad>"]

    @property
    def sos_id(self): return self.stoi["<sos>"]

    @property
    def eos_id(self): return self.stoi["<eos>"]


class LemmaPairs(Dataset):
    def __init__(self, path: Path, max_rows: int = 0):
        self.pairs: List[Tuple[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                w, lemma, *_ = line.rstrip("\n").split("\t")
                self.pairs.append((w, lemma))
                if max_rows and len(self.pairs) >= max_rows:
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def collate_fn(batch: List[Tuple[str, str]], vocab: CharVocab):
    ws, ls = zip(*batch)
    x = [vocab.encode(w) for w in ws]
    y = [vocab.encode(l) for l in ls]

    y_in = [[vocab.sos_id] + seq for seq in y]
    y_out = [seq + [vocab.eos_id] for seq in y]

    max_x = max(len(s) for s in x)
    max_y = max(len(s) for s in y_in)

    x_pad, x_len = [], []
    for s in x:
        x_len.append(len(s))
        x_pad.append(s + [vocab.pad_id] * (max_x - len(s)))

    y_in_pad, y_out_pad = [], []
    for a, b in zip(y_in, y_out):
        y_in_pad.append(a + [vocab.pad_id] * (max_y - len(a)))
        y_out_pad.append(b + [vocab.pad_id] * (max_y - len(b)))

    return (
        torch.tensor(x_pad, dtype=torch.long),
        torch.tensor(x_len, dtype=torch.long),
        torch.tensor(y_in_pad, dtype=torch.long),
        torch.tensor(y_out_pad, dtype=torch.long),
    )


class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hid, batch_first=True)
        self.dec = nn.LSTM(emb, hid, batch_first=True)
        self.proj = nn.Linear(hid, vocab_size)

    def forward(self, x, x_len, y_in):
        ex = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(ex, x_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.enc(packed)

        ey = self.emb(y_in)
        out, _ = self.dec(ey, (h, c))
        return self.proj(out)

    @torch.no_grad()
    def greedy(self, x, x_len, vocab: CharVocab, max_len: int = 32):
        ex = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(ex, x_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.enc(packed)

        B = x.size(0)
        cur = torch.full((B, 1), vocab.sos_id, device=x.device, dtype=torch.long)
        out_ids = [[] for _ in range(B)]

        state = (h, c)
        for _ in range(max_len):
            ey = self.emb(cur)
            dec_out, state = self.dec(ey, state)
            logits = self.proj(dec_out[:, -1, :])
            nxt = logits.argmax(-1)
            cur = nxt.unsqueeze(1)
            for i in range(B):
                out_ids[i].append(int(nxt[i]))

        res = []
        for seq in out_ids:
            cut = []
            for t in seq:
                if t == vocab.eos_id:
                    break
                cut.append(t)
            res.append(cut)
        return res


def exact_match(model: Seq2SeqLSTM, loader: DataLoader, vocab: CharVocab, device: torch.device, max_batches: int = 120):
    model.eval()
    ok = 0
    total = 0
    for bi, (x, x_len, y_in, y_out) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        x_len = x_len.to(device)
        y_out = y_out.to(device)

        pred_ids = model.greedy(x, x_len, vocab=vocab, max_len=y_out.size(1))
        for i in range(x.size(0)):
            gold = vocab.decode([t for t in y_out[i].tolist() if t not in (vocab.pad_id, vocab.eos_id)])
            pred = vocab.decode(pred_ids[i])
            total += 1
            ok += int(gold == pred)
    return ok / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = все строки")
    args = ap.parse_args()

    set_seed(args.seed)
    root = project_root()
    data_dir = root / "data" / "processed"
    model_dir = root / "models" / "lemma_lstm"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LemmaPairs(data_dir / "lemma_train.tsv", max_rows=args.max_train_rows)
    dev_ds = LemmaPairs(data_dir / "lemma_dev.tsv", max_rows=200000)

    vocab = CharVocab.build(train_ds.pairs)
    (model_dir / "vocab.json").write_text(json.dumps({"emb": args.emb, "hid": args.hid}, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqLSTM(vocab_size=len(vocab.itos), emb=args.emb, hid=args.hid).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for x, x_len, y_in, y_out in train_loader:
            x = x.to(device)
            x_len = x_len.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)

            logits = model(x, x_len, y_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        acc = exact_match(model, dev_loader, vocab, device)
        print(f"[epoch {ep}] loss={total_loss/max(steps,1):.4f} dev_exact={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), model_dir / "model.pt")

    print(f"Saved: {model_dir/'model.pt'} (best dev_exact={best:.4f})")


if __name__ == "__main__":
    main()
