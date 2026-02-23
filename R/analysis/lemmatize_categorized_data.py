#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lemmatize congressional speech text (by category+decade CSV) and cache lemma frequency tables by gender (+ POS).

Inputs:
  - CSV files matching: speech_categories_clean_LONG_<CATEGORY>_<DECADE>.csv
  - Each CSV must contain columns: text, gender   (gender values include 'M' and 'F')

Outputs:
  - A well-named lemma cache folder containing:
      <lemma_root>/
        manifest.json
        stopwords.txt
        <category>/<decade>/lemma_freq.parquet     # top 1000 (gender, lemma, pos) rows by count
                                                    # columns: gender, lemma, pos, N

Notes:
  - We save POS (Universal POS tags, spaCy token.pos_) because downstream scripts filter by POS.
  - Cache is "top 1000 rows per gender" AFTER grouping by (gender, lemma, pos).
    (So if POS splits a lemma across tags, you may have multiple rows for the same lemma.)

Parallelization:
  - Parallel across files using ProcessPoolExecutor with --workers
  - Each process loads spaCy once (safe)

Logging:
  - Main process prints a START and DONE line per file (category/decade), plus skip reasons
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import spacy


# ----------------------------
# Filename parsing
# ----------------------------
FNAME_RE = re.compile(r"^speech_categories_clean_LONG_(.*)_(\d{4})\.csv$")


def parse_meta(path: Path) -> Tuple[str, str]:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename format: {path.name}")
    return m.group(1), m.group(2)  # category, decade


# ----------------------------
# Stopwords (match your R logic)
# ----------------------------
def load_nltk_stopwords() -> List[str]:
    import nltk

    try:
        from nltk.corpus import stopwords

        return list(set(stopwords.words("english")))
    except Exception:
        nltk.download("stopwords")
        from nltk.corpus import stopwords

        return list(set(stopwords.words("english")))


def load_congress_stopwords(url: str) -> List[str]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = [ln.strip().lower() for ln in r.text.splitlines()]
        return sorted({ln for ln in lines if ln})
    except Exception as e:
        print(f"[WARN] Could not fetch congress stopwords from URL: {e}", file=sys.stderr)
        return []


def load_extra_stopwords(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln.strip().lower() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return sorted({ln for ln in lines if ln})


def build_stopwords(stopwords_extra: Optional[str]) -> List[str]:
    congress_url = "https://raw.githubusercontent.com/Democracy-Lab/feminized-language-of-democracy/main/analysis/congress_stopwords.csv"
    nltk_stops = [w.lower() for w in load_nltk_stopwords()]
    congress_stops = load_congress_stopwords(congress_url)
    extra = load_extra_stopwords(stopwords_extra)
    custom = ["mr", "us", "madam", "ms", "sir", "gentleman", "hr", "n't"]
    stop = sorted(set(nltk_stops) | set(congress_stops) | set(extra) | set(custom))
    return stop


# ----------------------------
# Cleaning (match your clean_word logic)
# ----------------------------
def clean_word(word: str, stopset: set) -> Optional[str]:
    if word is None:
        return None
    word = unicodedata.normalize("NFKD", str(word))
    cleaned = "".join(ch for ch in word if not unicodedata.category(ch).startswith("P")).strip()
    cleaned_lower = cleaned.lower()
    if not cleaned_lower or len(cleaned_lower) <= 1:
        return None
    if cleaned_lower in stopset:
        return None
    return cleaned_lower


# ----------------------------
# Document concatenation (mirror your R build_concat_docs)
# ----------------------------
def build_concat_docs(texts: List[str], max_chars: int) -> List[str]:
    docs: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in texts:
        if s is None:
            continue
        s = str(s)
        s_len = len(s)
        add_len = s_len if not cur else (1 + s_len)

        if cur_len > 0 and (cur_len + add_len) > max_chars:
            docs.append(" ".join(cur))
            cur = []
            cur_len = 0

        cur.append(s)
        cur_len += add_len

    if cur:
        docs.append(" ".join(cur))

    return docs


# ----------------------------
# Lemmatization + counts (per file)
# ----------------------------
def compute_lemma_freq_for_file(
    csv_path: str,
    lemma_root: str,
    stopwords: List[str],
    spacy_model: str,
    spacy_maxlen: int,
    concat_max_chars: int,
    top_cache_k: int = 1000,
    spacy_batch_size: int = 8,
) -> Dict[str, str]:
    """
    Compute lemma frequency by gender AND POS for one CSV and cache parquet.
    Returns dict with status metadata.
    """
    t0 = time.time()

    p = Path(csv_path)
    category, decade = parse_meta(p)

    out_dir = Path(lemma_root) / category / decade
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "lemma_freq.parquet"

    if out_file.exists():
        return {
            "file": str(p),
            "category": category,
            "decade": decade,
            "status": "SKIP_EXISTS",
            "cache": str(out_file),
            "elapsed_sec": f"{time.time() - t0:.3f}",
        }

    # read CSV
    try:
        df = pd.read_csv(p, usecols=["text", "gender"], encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        if "text" not in cols or "gender" not in cols:
            return {
                "file": str(p),
                "category": category,
                "decade": decade,
                "status": "SKIP_MISSING_COLS",
                "cache": str(out_file),
                "elapsed_sec": f"{time.time() - t0:.3f}",
            }
        df = df[[cols["text"], cols["gender"]]]
        df.columns = ["text", "gender"]

    df = df.dropna(subset=["text", "gender"])
    df["gender"] = df["gender"].astype(str)
    df = df[df["gender"].isin(["M", "F"])]

    if df.empty:
        return {
            "file": str(p),
            "category": category,
            "decade": decade,
            "status": "SKIP_EMPTY",
            "cache": str(out_file),
            "elapsed_sec": f"{time.time() - t0:.3f}",
        }

    m_texts = df.loc[df["gender"] == "M", "text"].astype(str).tolist()
    f_texts = df.loc[df["gender"] == "F", "text"].astype(str).tolist()
    if len(f_texts) == 0:
        return {
            "file": str(p),
            "category": category,
            "decade": decade,
            "status": "SKIP_NO_WOMEN",
            "cache": str(out_file),
            "rows": str(len(df)),
            "elapsed_sec": f"{time.time() - t0:.3f}",
        }

    stopset = set(stopwords)

    # spaCy init per-process
    # NOTE: do not disable tagger/parser/attribute_ruler/lemmatizer, since lemma_ depends on them
    nlp = spacy.load(spacy_model, disable=["ner"])
    nlp.max_length = spacy_maxlen

    # concat docs per gender
    docs_m = build_concat_docs(m_texts, concat_max_chars)
    docs_f = build_concat_docs(f_texts, concat_max_chars)

    # count by (gender, lemma, pos)
    counts: Dict[Tuple[str, str, str], int] = {}  # (gender, lemma, pos) -> N

    def consume_docs(docs: List[str], gender: str) -> None:
        for doc in nlp.pipe(docs, batch_size=spacy_batch_size):
            for tok in doc:
                lem_raw = tok.lemma_ if tok.lemma_ else tok.text
                lem = clean_word(lem_raw, stopset)
                if not lem:
                    continue
                pos = tok.pos_ or ""
                key = (gender, lem, pos)
                counts[key] = counts.get(key, 0) + 1

    consume_docs(docs_m, "M")
    consume_docs(docs_f, "F")

    if not counts:
        out_df = pd.DataFrame(columns=["gender", "lemma", "pos", "N"])
        out_df.to_parquet(out_file, index=False)
        return {
            "file": str(p),
            "category": category,
            "decade": decade,
            "status": "DONE_EMPTY_COUNTS",
            "cache": str(out_file),
            "rows": str(len(df)),
            "m_docs": str(len(docs_m)),
            "f_docs": str(len(docs_f)),
            "elapsed_sec": f"{time.time() - t0:.3f}",
        }

    out_rows = [{"gender": g, "lemma": l, "pos": pos, "N": n} for (g, l, pos), n in counts.items()]
    out_df = pd.DataFrame(out_rows)

    # keep top K rows per gender (after splitting by POS)
    out_df = out_df.sort_values(["gender", "N", "lemma", "pos"], ascending=[True, False, True, True])
    out_df = out_df.groupby("gender", as_index=False).head(top_cache_k)

    out_df.to_parquet(out_file, index=False)

    return {
        "file": str(p),
        "category": category,
        "decade": decade,
        "status": "DONE",
        "cache": str(out_file),
        "rows": str(len(df)),
        "m_docs": str(len(docs_m)),
        "f_docs": str(len(docs_f)),
        "elapsed_sec": f"{time.time() - t0:.3f}",
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing the CSV files")
    ap.add_argument("--stopwords_extra", default=None, help="Optional path to newline-separated extra stopwords")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers (processes)")
    ap.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy model name")
    ap.add_argument("--spacy_maxlen", type=int, default=4_000_000, help="spaCy nlp.max_length (characters)")
    ap.add_argument(
        "--concat_max_chars",
        type=int,
        default=None,
        help="Max chars per concatenated doc (default 0.75*maxlen capped at 1.5M)",
    )
    ap.add_argument("--spacy_batch_size", type=int, default=8, help="spaCy nlp.pipe batch_size")
    ap.add_argument("--top_n", type=int, default=50, help="Top-N for downstream (used only for naming/manifest)")
    ap.add_argument("--categories", nargs="*", default=None, help="Optional list of categories to include (exact match to filename token)")
    ap.add_argument("--output_parent", default="lemma_outputs", help="Parent directory to create the named lemma cache folder in")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.exists():
        raise SystemExit(f"input_dir does not exist: {in_dir}")

    # stopwords snapshot
    stopwords = build_stopwords(args.stopwords_extra)

    # discover files
    files = [p for p in in_dir.iterdir() if p.is_file() and FNAME_RE.match(p.name)]
    if args.categories:
        keep = set(args.categories)
        files = [p for p in files if parse_meta(p)[0] in keep]

    if not files:
        raise SystemExit("No matching files found.")

    # output folder naming
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    lemma_root = Path(args.output_parent) / f"lemma_cache_{stamp}_top{args.top_n}"
    lemma_root.mkdir(parents=True, exist_ok=True)

    (lemma_root / "stopwords.txt").write_text("\n".join(stopwords) + "\n", encoding="utf-8")

    # concat_max_chars default like your R: min(1.5M, floor(0.75 * spacy_maxlen))
    if args.concat_max_chars is None:
        concat_max_chars = min(1_500_000, int(math.floor(0.75 * args.spacy_maxlen)))
    else:
        concat_max_chars = int(args.concat_max_chars)

    manifest = {
        "created_at": stamp,
        "input_dir": str(in_dir),
        "top_n": args.top_n,
        "spacy_model": args.spacy_model,
        "spacy_maxlen": args.spacy_maxlen,
        "concat_max_chars": concat_max_chars,
        "spacy_batch_size": args.spacy_batch_size,
        "workers": args.workers,
        "stopwords_extra": args.stopwords_extra,
        "categories_filter": args.categories,
        "files": [str(p) for p in files],
        "results": [],
        "cache_format": {
            "file": "lemma_freq.parquet",
            "columns": ["gender", "lemma", "pos", "N"],
            "top_cache_k_per_gender": 1000,
            "pos": "spaCy token.pos_ (Universal POS tags)",
        },
    }

    print(f"[INFO] lemma_root = {lemma_root}")
    print(f"[INFO] files = {len(files)} | workers = {args.workers}")
    print(f"[INFO] spacy_maxlen = {args.spacy_maxlen} | concat_max_chars = {concat_max_chars}")

    # Submit work
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for p in files:
            cat, dec = parse_meta(p)
            print(f"[START] category={cat} decade={dec} file={p.name}")
            fut = ex.submit(
                compute_lemma_freq_for_file,
                str(p),
                str(lemma_root),
                stopwords,
                args.spacy_model,
                int(args.spacy_maxlen),
                int(concat_max_chars),
                1000,
                int(args.spacy_batch_size),
            )
            futs[fut] = (p, cat, dec)

        for fut in as_completed(futs):
            p, cat, dec = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                # record failure but keep going
                res = {
                    "file": str(p),
                    "category": cat,
                    "decade": dec,
                    "status": "ERROR",
                    "error": repr(e),
                }

            manifest["results"].append(res)
            status = res.get("status", "UNKNOWN")
            elapsed = res.get("elapsed_sec", "NA")
            cache = res.get("cache", str(lemma_root / cat / dec / "lemma_freq.parquet"))
            rows = res.get("rows", "NA")
            print(f"[DONE]  category={cat} decade={dec} status={status} rows={rows} elapsed_sec={elapsed} cache={cache}")

    (lemma_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Wrote lemma cache folder:")
    print(str(lemma_root))


if __name__ == "__main__":
    main()
