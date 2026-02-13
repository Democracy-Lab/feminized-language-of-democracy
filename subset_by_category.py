# subset_by_category.py

from __future__ import annotations

import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# CLI thresholds (match R: --thr=0.03 or --thr 0.03)
# ──────────────────────────────────────────────────────────────
def get_cli_opt(name: str, default: str | None = None) -> str | None:
    a = sys.argv[1:]
    # --name=value
    for arg in a:
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
    # --name value
    for i, arg in enumerate(a):
        if arg == f"--{name}" and i + 1 < len(a):
            return a[i + 1]
    return default


DEFAULT_THR = float(get_cli_opt("thr", "0.03"))
DEFAULT_POM = float(get_cli_opt("pom", "0.85"))


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
VOCAB_URL = os.getenv(
    "CONGRESS_VOCAB_URL",
    "https://raw.githubusercontent.com/stephbuon/text-mine-congress/refs/heads/main/analysis/congress_controlled_vocab.csv",
)

MIN_TOKEN_NCHAR = int(os.getenv("MIN_TOKEN_NCHAR", "1"))
DATE_COL = "date"
KW_BATCH_SIZE = int(os.getenv("KW_BATCH_SIZE", "64"))

DECADE_TEST = False
DECADE_SELECTION = 2010


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def safelabel(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))


def str_squish(x: str) -> str:
    return " ".join(str(x).split())


def escape_kw(x: str) -> str:
    """
    R:
      x <- str_squish(x)
      rx <- str_replace_all(x, "([.^$|()\\[\\]{}*+?\\\\-])", "\\\\\\1")
      paste0("\\b", rx, "(e?s)?\\b")
    """
    x = str_squish(x)
    rx = re.sub(r"([.^$|()\[\]{}*+?\\\-])", r"\\\1", x)
    return rf"\b{rx}(e?s)?\b"


def build_day_key(df: pd.DataFrame) -> pd.Series:
    """
    Compatibility fix (glaring issue B):
    - Debate grouping must come from 'date' (not parsing source_file).
    """
    if DATE_COL not in df.columns:
        raise ValueError(
            f"Missing required column '{DATE_COL}'. "
            "Debate grouping is defined by consecutive speeches on the same day."
        )
    return df[DATE_COL].astype(str)


def load_vocab_cached(vocab_url: str, cache_path: Path) -> pd.DataFrame:
    """
    Mirrors the R caching + 3 retries + backoff.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        ok = False
        for i in range(1, 4):
            try:
                urlretrieve(vocab_url, cache_path)
                ok = cache_path.exists() and cache_path.stat().st_size > 0
            except Exception:
                ok = False

            if ok:
                break
            time.sleep(1.0 * i)

        if not ok:
            raise RuntimeError(
                "Failed to fetch controlled vocabulary after retries. "
                f"Tried URL: {vocab_url}\n"
                f"You can place a copy at: {cache_path} and rerun."
            )

    return pd.read_csv(cache_path)


def _batches(n: int, batch_size: int) -> list[np.ndarray]:
    idx = np.arange(n)
    return np.array_split(idx, math.ceil(n / max(batch_size, 1)))


def _stable_sort_kind() -> str:
    # mergesort is stable
    return "mergesort"


# ──────────────────────────────────────────────────────────────
# Core: debate-level categorization via date + title + sequential runs
# ──────────────────────────────────────────────────────────────
def subset_by_category(
    f: pd.DataFrame,
    decade: str | int,
    categories_dir: str | Path,
    *,
    thr: float = DEFAULT_THR,
    pom: float = DEFAULT_POM,
    vocab_url: str = VOCAB_URL,
    kw_batch_size: int = KW_BATCH_SIZE,
    cleaned_base_dir: Path = Path("data") / "gender_analysis",
) -> None:
    """
    Compatible with the new parse_congress.py + pipeline:

    - Uses stable identifiers:
        * speech_id (unique per speech / input row)
        * speech_order (preserves original speech order within the decade file)

    - Defines debates as:
        * within each day_key (=date), consecutive speeches with the same normalized title_key

    - Glaring issue fixes implemented:
        A) Build the debate map ONCE and reuse it (no second pass / no divergence).
        B) day_key comes from 'date' (not parsing source_file).

    Expects `f` to be CLEANED token-level data for the decade (already stopwords-filtered).

    Required columns in f:
      - token
      - speech_id
      - speech_order
      - title
      - date
      - debate grouping is sequential, so correct ordering must be possible.

    Optional (kept if present, used in speech-level outputs):
      - gender, state, party, source_file, doc_id, piece_id, sentence_id, token_id
    """
    decade = str(decade)

    if DECADE_TEST and int(decade) != int(DECADE_SELECTION):
        print(f"Skipping decade {decade} (DECADE_TEST enabled; running only {DECADE_SELECTION})")
        return

    print(f"== Categorizing decade {decade} at DEBATE level (date + title + sequential runs) ==")

    # Output dirs
    tag = f"thr{thr:.3f}_pom{pom:.2f}"
    categories_dir = Path(categories_dir)

    score_dir = categories_dir / f"tfidf_norm_scores_{tag}"
    assign_dir = categories_dir / f"tfidf_norm_assignments_{tag}"
    filtered_tokens_dir = categories_dir / f"tfidf_norm_filtered_tokens_{tag}"
    qc_dir = categories_dir / f"tfidf_norm_qc_{tag}"
    speech_clean_text_dir = categories_dir / f"speech_level_clean_text_{tag}"

    for d in [score_dir, assign_dir, filtered_tokens_dir, qc_dir, speech_clean_text_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────
    # Validate + prep tokens
    # ───────────────────────────────────────────────────────────
    required = {"token", "speech_id", "speech_order", "title", DATE_COL}
    missing = required - set(f.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    f = f.copy()
    f["token"] = f["token"].astype(str).str.lower()

    # Skip speeches with missing/empty title (title required for debate grouping)
    total_speeches_before = f["speech_id"].nunique(dropna=True)
    f = f[f["title"].notna() & (f["title"].astype(str).str.strip().str.len() > 0)].copy()
    total_speeches_after = f["speech_id"].nunique(dropna=True)
    skipped_count = total_speeches_before - total_speeches_after

    # Normalize title + build day_key
    f["title_key"] = f["title"].astype(str).str.lower().map(str_squish)
    f["day_key"] = build_day_key(f)

    # ───────────────────────────────────────────────────────────
    # Build debate map ONCE (glaring issue A)
    # Debate = sequential run of same title_key within the same day_key
    # Ordering uses speech_order (not source_file/doc parsing)
    # ───────────────────────────────────────────────────────────
    speech_tbl = (
        f[["speech_id", "speech_order", "title", "title_key", "day_key"]]
        .drop_duplicates()
        .sort_values(["day_key", "speech_order"], kind=_stable_sort_kind())
        .reset_index(drop=True)
    )

    speech_tbl["prev_title_key"] = speech_tbl.groupby("day_key")["title_key"].shift(1)
    title_changed = (speech_tbl["title_key"] != speech_tbl["prev_title_key"]).fillna(True)
    speech_tbl["run_id"] = title_changed.groupby(speech_tbl["day_key"]).cumsum().astype(int)

    speech_tbl["debate_id"] = (
        speech_tbl["day_key"].astype(str)
        + "__"
        + speech_tbl["title_key"].astype(str)
        + "__"
        + speech_tbl["run_id"].astype(str)
    )

    # Attach debate_id to tokens (single source of truth)
    f = f.merge(speech_tbl[["speech_id", "debate_id"]], on="speech_id", how="left")

    # QC debate map
    debate_map = speech_tbl[["debate_id", "day_key", "title", "run_id"]].drop_duplicates()
    debate_map.to_csv(qc_dir / f"debate_map_{decade}.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # Debate text + word counts (from CLEANED tokens)
    # ───────────────────────────────────────────────────────────
    # For word_count, count whitespace-separated tokens in joined text (as in your prior script)
    debate_tokens = (
        f.groupby("debate_id", sort=False)["token"]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .reset_index(name="text")
    )
    debate_tokens["word_count"] = debate_tokens["text"].str.count(r"\S+", flags=re.IGNORECASE)

    MIN_WORDS = 100
    dropped_n = int((debate_tokens["word_count"] < MIN_WORDS).sum())
    n_before = len(debate_tokens)
    pct = (100.0 * dropped_n / n_before) if n_before else 0.0
    print(f"Dropping {dropped_n} debates with <{MIN_WORDS} words out of {n_before} ({pct:.1f}%).")
    debate_tokens = debate_tokens[debate_tokens["word_count"] >= MIN_WORDS].copy()

    n_debates = len(debate_tokens)
    if n_debates == 0:
        warnings.warn("No debates found after filtering; skipping.")
        print(f"Speeches not grouped to debate (missing/empty title): {skipped_count}")
        print(f"Speeches grouped to debate and used: {total_speeches_after}")
        return

    # For TF normalization, use word_count (debate-level total words)
    deb_ids = debate_tokens["debate_id"].to_numpy()
    deb_text = debate_tokens["text"]
    deb_words = np.maximum(debate_tokens["word_count"].to_numpy(dtype=float), 1.0)

    # ───────────────────────────────────────────────────────────
    # Controlled vocab (cached)
    # ───────────────────────────────────────────────────────────
    vocab_cache = cleaned_base_dir / "congress_controlled_vocab.csv"
    vocab = load_vocab_cached(vocab_url, vocab_cache)

    if "Category" not in vocab.columns or "Keywords" not in vocab.columns:
        raise ValueError("Controlled vocab CSV must have columns: Category, Keywords")

    vocab = vocab[["Category", "Keywords"]].copy()
    vocab["Keywords"] = vocab["Keywords"].astype(str).str.split(r",\s*")
    vocab = vocab.explode("Keywords").rename(columns={"Category": "category", "Keywords": "keyword"})
    vocab["keyword"] = vocab["keyword"].astype(str).map(str_squish).str.lower()
    vocab["pattern"] = vocab["keyword"].map(escape_kw)
    vocab = vocab[["category", "keyword", "pattern"]].copy()

    cat_levels = sorted(vocab["category"].dropna().unique().tolist())
    cat_index = {c: i for i, c in enumerate(cat_levels)}

    # ───────────────────────────────────────────────────────────
    # Two-pass TF–IDF (batched)
    # ───────────────────────────────────────────────────────────
    print("Pass 1/2: computing DF per keyword (batched) ...")
    K = len(vocab)
    df_counts = np.zeros(K, dtype=float)
    batches = _batches(K, kw_batch_size)

    for idx in batches:
        sub = vocab.iloc[idx]
        pats = sub["pattern"].tolist()
        for j, pat in enumerate(pats):
            counts = deb_text.str.count(pat, flags=re.IGNORECASE)
            df_counts[idx[j]] = float((counts > 0).sum())

    idf = np.log((n_debates + 1.0) / (df_counts + 1.0)) + 1.0

    print("Pass 2/2: accumulating category scores (batched) ...")
    cat_mat = np.zeros((n_debates, len(cat_levels)), dtype=float)

    for idx in batches:
        sub = vocab.iloc[idx]
        pats = sub["pattern"].tolist()
        cats = sub["category"].tolist()
        idfs = idf[idx]

        for j, (pat, cat) in enumerate(zip(pats, cats)):
            counts = deb_text.str.count(pat, flags=re.IGNORECASE).to_numpy(dtype=float)
            if not np.any(counts):
                continue
            norm_tf_1k = (counts / deb_words) * 1000.0
            contrib = norm_tf_1k * float(idfs[j])
            col = cat_index.get(cat)
            if col is not None:
                cat_mat[:, col] += contrib

    # Long scores table
    cat_scores = pd.DataFrame({"debate_id": deb_ids})
    for i, cname in enumerate(cat_levels):
        cat_scores[cname] = cat_mat[:, i]

    cat_scores = cat_scores.melt(id_vars=["debate_id"], var_name="category", value_name="category_score")
    cat_scores = cat_scores.merge(
        pd.DataFrame({"debate_id": deb_ids, "total_words": deb_words.astype(float)}),
        on="debate_id",
        how="left",
    )

    # Persist scores
    score_parquet = score_dir / f"category_scores_{decade}_unit-debate_tfidf_norm_1k.parquet"
    score_csv = score_dir / f"category_scores_{decade}_unit-debate_tfidf_norm_1k.csv"
    cat_scores.to_parquet(score_parquet, index=False)
    cat_scores.to_csv(score_csv, index=False)

    # Assignments
    print("Assigning categories (multi-label) ...")
    max_score = cat_scores.groupby("debate_id")["category_score"].transform("max")
    include = (cat_scores["category_score"] >= thr) & (cat_scores["category_score"] >= pom * max_score)
    assignments = cat_scores.loc[include].copy().sort_values(["debate_id", "category_score"], ascending=[True, False])

    assign_parquet = assign_dir / f"assignments_{decade}_unit-debate_thr{thr:.3f}_pom{pom:.2f}.parquet"
    assign_csv = assign_dir / f"assignments_{decade}_unit-debate_thr{thr:.3f}_pom{pom:.2f}.csv"
    assignments.to_parquet(assign_parquet, index=False)
    assignments.to_csv(assign_csv, index=False)

    # ───────────────────────────────────────────────────────────
    # Per-category outputs + CLEAN speech-level CSVs with TEXT
    # (uses the SAME debate_id map already attached to f)
    # ───────────────────────────────────────────────────────────
    print("Preparing CLEANED per-category outputs (tokens + speech-level CSV) ...")

    # Build speech-level clean text (one row per speech) using deterministic token ordering when possible
    base_group_cols = ["speech_id", "debate_id", "speech_order", "gender", "date", "state", "party", "title"]
    present_group_cols = [c for c in base_group_cols if c in f.columns]
    if "speech_id" not in present_group_cols or "debate_id" not in present_group_cols:
        raise ValueError("Internal error: speech_id/debate_id missing from cleaned tokens.")

    # Determine best available token-order columns for reconstruction
    order_cols = [c for c in ["speech_id", "piece_id", "sentence_id", "token_id"] if c in f.columns]
    if order_cols:
        f_for_text = f.sort_values(order_cols, kind=_stable_sort_kind()).copy()
    else:
        # fallback: keep existing row order (not ideal, but pipeline from scratch should include token ids)
        f_for_text = f.copy()

    clean_speech_text = (
        f_for_text.groupby(present_group_cols, dropna=False, sort=False)["token"]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .reset_index(name="text")
    )

    cats = sorted(assignments["category"].dropna().unique().tolist())
    for cat in cats:
        keep_debates = assignments.loc[assignments["category"] == cat, ["debate_id"]].drop_duplicates()
        keep_set = set(keep_debates["debate_id"].tolist())
        safe_cat = safelabel(cat)

        # Token subset per category
        out_tokens = f[f["debate_id"].isin(keep_set)].copy()
        out_tokens.to_parquet(
            filtered_tokens_dir / f"{safe_cat}_{decade}_unit-debate_tfidf_norm_1k_filtered_tokens.parquet",
            index=False,
        )

        # CLEAN speech-level CSV with TEXT
        clean_long = clean_speech_text[clean_speech_text["debate_id"].isin(keep_set)].copy()
        clean_long["category"] = cat

        # Keep a consistent column order (only include those that exist)
        wanted = [
            "speech_id",
            "debate_id",
            "category",
            "speech_order",
            "gender",
            "date",
            "title",
            "state",
            "party",
            "text",
        ]
        existing = [c for c in wanted if c in clean_long.columns]
        clean_long = clean_long[existing]

        clean_long.to_csv(
            speech_clean_text_dir / f"speech_categories_clean_LONG_{safe_cat}_{decade}.csv",
            index=False,
        )

    print(f"Speeches not grouped to debate (missing/empty title): {skipped_count}")
    print(f"Speeches grouped to debate and used: {total_speeches_after}")
    print("subset_by_category complete (debate-level with sequential runs, titles required).")


if __name__ == "__main__":
    print("Loaded subset_by_category (debate-level with sequential runs, uses speech_id + speech_order).")
