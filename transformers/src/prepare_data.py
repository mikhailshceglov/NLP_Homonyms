from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

from utils import ensure_dir, project_root, set_seed


def first_alpha_grammeme(node) -> Optional[str]:
    for g in node.findall(".//g"):
        v = g.get("v")
        if v and v.isalpha():
            return v
    return None


def split_writer(
    seed: int,
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    p_dev: float = 0.05,
    p_test: float = 0.05,
):
    rnd = random.Random(seed)

    f_train = train_path.open("w", encoding="utf-8")
    f_dev = dev_path.open("w", encoding="utf-8")
    f_test = test_path.open("w", encoding="utf-8")

    def write(obj: Dict):
        r = rnd.random()
        if r < p_test:
            f = f_test
        elif r < p_test + p_dev:
            f = f_dev
        else:
            f = f_train
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close():
        f_train.close()
        f_dev.close()
        f_test.close()

    return write, close


def export_pos_from_annot(annot_xml: Path, out_dir: Path, seed: int) -> Path:
    ensure_dir(out_dir)
    train_p = out_dir / "pos_train.jsonl"
    dev_p = out_dir / "pos_dev.jsonl"
    test_p = out_dir / "pos_test.jsonl"
    tagset_p = out_dir / "pos_tagset.json"

    write_row, close_writers = split_writer(seed, train_p, dev_p, test_p)
    tagset: Set[str] = set()

    tokens: List[str] = []
    tags: List[str] = []

    for event, elem in ET.iterparse(str(annot_xml), events=("start", "end")):
        if event == "start" and elem.tag == "sentence":
            tokens = []
            tags = []
        elif event == "end" and elem.tag == "token":
            tok_text = elem.get("text")
            if tok_text:
                pos = None
                for l in elem.findall(".//l"):
                    pos = first_alpha_grammeme(l)
                    if pos:
                        break
                if pos:
                    tokens.append(tok_text)
                    tags.append(pos)
                    tagset.add(pos)
            elem.clear()
        elif event == "end" and elem.tag == "sentence":
            if tokens:
                write_row({"tokens": tokens, "tags": tags})
            elem.clear()

    close_writers()
    with tagset_p.open("w", encoding="utf-8") as f:
        json.dump(sorted(tagset), f, ensure_ascii=False, indent=2)

    return tagset_p


def export_lemma_pairs_from_dict(
    dict_xml: Path, out_dir: Path, seed: int, max_per_lemma: int = 0
) -> None:
    ensure_dir(out_dir)
    train_p = out_dir / "lemma_train.tsv"
    dev_p = out_dir / "lemma_dev.tsv"
    test_p = out_dir / "lemma_test.tsv"

    rnd = random.Random(seed)

    f_train = train_p.open("w", encoding="utf-8")
    f_dev = dev_p.open("w", encoding="utf-8")
    f_test = test_p.open("w", encoding="utf-8")

    def pick_file():
        r = rnd.random()
        if r < 0.05:
            return f_test
        if r < 0.10:
            return f_dev
        return f_train

    for event, elem in ET.iterparse(str(dict_xml), events=("end",)):
        if elem.tag != "lemma":
            continue

        l = elem.find("l")
        if l is None:
            elem.clear()
            continue

        lemma_txt = l.get("t")
        if not lemma_txt:
            elem.clear()
            continue

        pos_lemma = first_alpha_grammeme(l) or ""

        count = 0
        for f in elem.findall("f"):
            form = f.get("t")
            if not form:
                continue
            pos_form = first_alpha_grammeme(f) or pos_lemma
            out = pick_file()
            out.write(f"{form}\t{lemma_txt}\t{pos_form}\n")
            count += 1
            if max_per_lemma and count >= max_per_lemma:
                break

        elem.clear()

    f_train.close()
    f_dev.close()
    f_test.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--max-forms-per-lemma",
        type=int,
        default=0,
        help="0 = без лимита (много данных), иначе ограничивает число форм на лемму",
    )
    args = ap.parse_args()
    set_seed(args.seed)

    root = project_root()
    raw_dir = Path(args.raw_dir) if args.raw_dir else (root / "data" / "raw")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "data" / "processed")

    annot_xml = raw_dir / "annot.opcorpora.xml"
    dict_xml = raw_dir / "dict.opcorpora.xml"

    print("1) Export POS dataset...")
    tagset_path = export_pos_from_annot(annot_xml, out_dir, seed=args.seed)
    print(f"   tagset: {tagset_path}")

    print("2) Export lemma pairs...")
    export_lemma_pairs_from_dict(
        dict_xml, out_dir, seed=args.seed, max_per_lemma=args.max_forms_per_lemma
    )
    print("Done.")


if __name__ == "__main__":
    main()
