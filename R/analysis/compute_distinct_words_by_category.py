#!/usr/bin/env python3
"""
Compute distinctive words using log-likelihood for categorized congressional speeches.
Uses pre-computed lemmatized frequency data from cached lemma frequency files.

BACKWARD COMPATIBILITY:
- Supports BOTH:
    {lemma_dir}/{category}/{decade}/lemma_freq.parquet   (preferred)
    {lemma_dir}/{category}/{decade}/lemma_freq.rds       (legacy)

Expected columns:
- Required: gender, lemma, N
- Optional: pos  (for POS filtering)
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Optional: RDS support (legacy)
try:
    import pyreadr  # type: ignore
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False

# Optional: Parquet support via pyarrow
try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# For stopwords
try:
    from nltk.corpus import stopwords as nltk_stopwords
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    raise ImportError("nltk is required. Install with: pip install nltk")


def load_congress_stopwords() -> set:
    """
    Load congress-specific stopwords from GitHub repository.
    Returns a set of stopwords.
    """
    url = "https://raw.githubusercontent.com/Democracy-Lab/feminized-language-of-democracy/main/analysis/congress_stopwords.csv"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        stopwords = set()
        for line in response.text.strip().split("\n"):
            word = line.strip()
            if word:
                stopwords.add(word.lower())

        return stopwords
    except Exception as e:
        warnings.warn(f"Failed to load congress stopwords from GitHub: {e}")
        return set()


def initialize_stopwords() -> set:
    """
    Initialize combined stopword list from congress stopwords and NLTK English stopwords.
    Returns a set of stopwords.
    """
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords", quiet=True)

    nltk_stops = set(nltk_stopwords.words("english"))
    congress_stops = load_congress_stopwords()
    combined_stops = nltk_stops | congress_stops

    print(f"Loaded {len(nltk_stops)} NLTK stopwords")
    print(f"Loaded {len(congress_stops)} congress stopwords")
    print(f"Total unique stopwords: {len(combined_stops)}")

    return combined_stops


def clean_word(word: str, stopwords_set: set) -> Optional[str]:
    word = unicodedata.normalize("NFKD", word)

    cleaned = "".join(
        char for char in word
        if not unicodedata.category(char).startswith("P")
    ).strip()

    cleaned_lower = cleaned.lower()

    if cleaned_lower in stopwords_set:
        return None

    if len(cleaned_lower) <= 1:
        return None

    return cleaned_lower


def discover_category_decade_files(lemma_dir: Path) -> pd.DataFrame:
    """
    Discover all cached lemma files and extract category/decade metadata.

    Expected structure:
      {lemma_dir}/{category}/{decade}/lemma_freq.parquet   (preferred)
      {lemma_dir}/{category}/{decade}/lemma_freq.rds       (legacy)

    Returns DataFrame with columns: category, decade
    """
    parquet_files = list(lemma_dir.glob("*/*/lemma_freq.parquet"))
    rds_files = list(lemma_dir.glob("*/*/lemma_freq.rds"))

    files = parquet_files + rds_files
    if not files:
        raise ValueError(
            f"No lemma_freq.parquet or lemma_freq.rds files found in: {lemma_dir}"
        )

    metadata = []
    for f in files:
        decade = f.parent.name
        category = f.parent.parent.name

        try:
            decade_int = int(decade)
            metadata.append({"category": category, "decade": decade_int})
        except ValueError:
            warnings.warn(f"Skipping invalid decade in path: {f}")

    return (
        pd.DataFrame(metadata)
        .drop_duplicates()
        .sort_values(["category", "decade"])
        .reset_index(drop=True)
    )


def _read_parquet(cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Read parquet cache using pandas.read_parquet (pyarrow or fastparquet backend).
    """
    try:
        df = pd.read_parquet(cache_path)
        return df
    except Exception as e:
        warnings.warn(f"Failed to read parquet cache {cache_path}: {e}")
        return None


def _read_rds(cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Read legacy RDS cache using pyreadr (if available).
    """
    if not HAS_PYREADR:
        warnings.warn(
            f"Found legacy RDS cache but pyreadr is not installed: {cache_path}"
        )
        return None

    try:
        result = pyreadr.read_r(str(cache_path))
        df = result[None] if None in result else result[list(result.keys())[0]]
        return df
    except Exception as e:
        warnings.warn(f"Failed to read RDS cache {cache_path}: {e}")
        return None


def load_lemma_cache(lemma_dir: Path, category: str, decade: str) -> Optional[pd.DataFrame]:
    """
    Load cached lemmatized frequency data.
    Prefers parquet if present; otherwise falls back to RDS.

    Returns DataFrame with columns:
      - required: gender, lemma, N
      - optional: pos
    """
    parquet_path = lemma_dir / category / str(decade) / "lemma_freq.parquet"
    rds_path = lemma_dir / category / str(decade) / "lemma_freq.rds"

    df: Optional[pd.DataFrame] = None
    if parquet_path.exists():
        df = _read_parquet(parquet_path)
    elif rds_path.exists():
        df = _read_rds(rds_path)
    else:
        warnings.warn(f"No cache file found for category={category} decade={decade} under {lemma_dir}")
        return None

    if df is None or df.empty:
        return None

    required_cols = ["gender", "lemma", "N"]
    if not all(col in df.columns for col in required_cols):
        warnings.warn(
            f"Cache file missing expected columns ({required_cols}). "
            f"Found: {list(df.columns)} | category={category} decade={decade}"
        )
        return None

    # Ensure correct types
    df = df.copy()
    df["N"] = pd.to_numeric(df["N"], errors="coerce").fillna(0).astype(int)
    df["lemma"] = df["lemma"].astype(str)
    df["gender"] = df["gender"].astype(str)

    if "pos" in df.columns:
        df["pos"] = df["pos"].astype(str)

    return df


def get_word_counts_from_cache(
    cache_df: pd.DataFrame,
    gender: str,
    stopwords_set: set,
    pos_filter: Optional[List[str]] = None,
) -> Tuple[Counter, int]:
    """
    Extract word counts from cached lemma frequency data.
    Applies cleaning: removes stopwords, punctuation, and single-character tokens.

    Args:
        cache_df: DataFrame with columns gender, lemma, pos (optional), N
        gender: "M" or "F"
        stopwords_set: Set of stopwords to filter out
        pos_filter: Optional list of POS tags to include (e.g., ['NOUN', 'VERB'])

    Returns:
        (Counter of words, total word count)
    """
    gender_data = cache_df[cache_df["gender"] == gender].copy()
    if len(gender_data) == 0:
        return Counter(), 0

    # Filter by POS if specified
    if pos_filter is not None:
        if "pos" not in gender_data.columns:
            warnings.warn("POS filter requested but cache doesn't contain 'pos' column")
            return Counter(), 0
        gender_data = gender_data[gender_data["pos"].isin(pos_filter)]
    else:
        # NO POS filter: aggregate across all POS tags for each lemma
        if "pos" in gender_data.columns:
            gender_data = gender_data.groupby("lemma", as_index=False)["N"].sum()

    cleaned_counts = Counter()
    for word, count in zip(gender_data["lemma"], gender_data["N"]):
        cleaned_word = clean_word(word, stopwords_set)
        if cleaned_word is not None:
            cleaned_counts[cleaned_word] += int(count)

    total = sum(cleaned_counts.values())
    return cleaned_counts, total


def compute_log_likelihood(
    target_counts: Counter,
    target_total: int,
    reference_counts: Counter,
    reference_total: int,
    top_n: int,
) -> pd.DataFrame:
    """
    Compute log-likelihood G² for words in target vs reference corpus.
    Returns top N words sorted by G² score.
    Only includes words that are OVERREPRESENTED in target (not reference).
    """
    if target_total == 0 or reference_total == 0:
        return pd.DataFrame(columns=["feature", "G2", "p"])

    words = set(target_counts.keys()) | set(reference_counts.keys())
    results = []

    for word in words:
        a = target_counts.get(word, 0)
        b = reference_counts.get(word, 0)
        c = target_total - a
        d = reference_total - b

        n = a + b + c + d
        if n == 0:
            continue

        e_a = (a + b) * (a + c) / n
        e_b = (a + b) * (b + d) / n
        e_c = (c + d) * (a + c) / n
        e_d = (c + d) * (b + d) / n

        if a < e_a:
            continue

        g2 = 0.0
        for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]:
            if obs > 0 and exp > 0:
                g2 += 2 * obs * math.log(obs / exp)

        results.append({"feature": word, "G2": g2, "p": None})

    df = pd.DataFrame(results)
    if len(df) == 0:
        return pd.DataFrame(columns=["feature", "G2", "p"])

    return df.nlargest(top_n, "G2").reset_index(drop=True)


def aggregate_counts_across_files(
    lemma_dir: Path,
    file_list: List[Tuple[str, int]],
    gender: str,
    stopwords_set: set,
    pos_filter: Optional[List[str]] = None,
) -> Tuple[Counter, int]:
    """
    Aggregate word counts across multiple category/decade files for a given gender.
    """
    aggregated = Counter()

    for category, decade in file_list:
        cache_df = load_lemma_cache(lemma_dir, category, str(decade))
        if cache_df is None:
            continue

        counts, _ = get_word_counts_from_cache(cache_df, gender, stopwords_set, pos_filter)
        aggregated.update(counts)

    total = sum(aggregated.values())
    return aggregated, total


def main():
    parser = argparse.ArgumentParser(
        description="Compute distinctive words using pre-computed lemmatized data"
    )
    parser.add_argument(
        "--lemma_dir",
        required=True,
        help="Directory containing cached lemmatized data ({category}/{decade}/lemma_freq.parquet or .rds)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: auto-generated based on top_n, pos, and date)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top distinctive words to extract (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--pos",
        nargs="+",
        default=None,
        help="Optional POS tags to filter (e.g., NOUN ADJ VERB). If not set, uses all words.",
    )

    args = parser.parse_args()

    lemma_dir = Path(args.lemma_dir)
    top_n = args.top
    n_workers = args.workers
    pos_filter = args.pos

    if args.output is None:
        date_str = datetime.now().strftime("%m-%d-%y")
        if pos_filter:
            pos_str = "_".join(pos_filter).lower()
            output_dir = Path(f"distinct_{pos_str}s_by_category_decade_gender_top{top_n}_{date_str}")
        else:
            output_dir = Path(f"distinct_words_by_category_decade_gender_top{top_n}_{date_str}")
    else:
        output_dir = Path(args.output)

    if not lemma_dir.exists():
        raise ValueError(f"Lemma directory does not exist: {lemma_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_category").mkdir(exist_ok=True)
    (output_dir / "by_decade").mkdir(exist_ok=True)
    (output_dir / "by_decade_and_category").mkdir(exist_ok=True)
    (output_dir / "by_gender_and_category").mkdir(exist_ok=True)
    (output_dir / "by_gender_and_decade").mkdir(exist_ok=True)
    (output_dir / "by_gender_decade_category").mkdir(exist_ok=True)

    if pos_filter:
        file_suffix = "_".join(pos_filter).lower()
    else:
        file_suffix = "words"

    print("=" * 70)
    print(f"Top N: {top_n}")
    print(f"POS filter: {pos_filter if pos_filter else 'None (all words)'}")
    print(f"Lemma directory: {lemma_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {n_workers}")
    print(f"Parquet support (pyarrow present): {HAS_PYARROW}")
    print(f"Legacy RDS support (pyreadr present): {HAS_PYREADR}")
    print("=" * 70)
    print()

    # ── Initialize stopwords ────────────────────────────────────────────
    print("Initializing stopwords...")
    stopwords_set = initialize_stopwords()
    print()

    # ── Discover category/decade combinations from lemma cache ────────────
    print("Discovering cached lemma files...")
    metadata_df = discover_category_decade_files(lemma_dir)

    print(f"Found {len(metadata_df)} category/decade combinations")

    categories = sorted(metadata_df["category"].unique())
    decades = sorted(metadata_df["decade"].unique())

    print(f"\nCategories ({len(categories)}): {', '.join(categories)}")
    print(f"Decades ({len(decades)}): {', '.join(map(str, decades))}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 1) By Category
    # ══════════════════════════════════════════════════════════════════════
    print("[1/6] Computing distinctive words per category...")

    def process_category(cat):
        cat_files = metadata_df[metadata_df["category"] == cat][["category", "decade"]].values.tolist()
        other_files = metadata_df[metadata_df["category"] != cat][["category", "decade"]].values.tolist()
        if len(cat_files) == 0:
            return None

        target_m, _ = aggregate_counts_across_files(lemma_dir, cat_files, "M", stopwords_set, pos_filter)
        target_f, _ = aggregate_counts_across_files(lemma_dir, cat_files, "F", stopwords_set, pos_filter)
        target_all = target_m + target_f
        target_total = sum(target_all.values())

        ref_m, _ = aggregate_counts_across_files(lemma_dir, other_files, "M", stopwords_set, pos_filter)
        ref_f, _ = aggregate_counts_across_files(lemma_dir, other_files, "F", stopwords_set, pos_filter)
        ref_all = ref_m + ref_f
        ref_total = sum(ref_all.values())

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_all, target_total, ref_all, ref_total, top_n)
        results["category"] = cat
        return results

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_category)(cat) for cat in categories
    )

    cat_results = [r for r in results if r is not None]
    if cat_results:
        cat_results_df = pd.concat(cat_results, ignore_index=True)
        cat_results_df = cat_results_df[["category", "feature", "G2", "p"]]

        output_path = output_dir / "by_category" / f"distinctive_{file_suffix}_by_category_top{top_n}.csv"
        cat_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    # ══════════════════════════════════════════════════════════════════════
    # 2) By Decade
    # ══════════════════════════════════════════════════════════════════════
    print("[2/6] Computing distinctive words per decade...")

    def process_decade(dec):
        dec_files = metadata_df[metadata_df["decade"] == dec][["category", "decade"]].values.tolist()
        other_files = metadata_df[metadata_df["decade"] != dec][["category", "decade"]].values.tolist()
        if len(dec_files) == 0:
            return None

        target_m, _ = aggregate_counts_across_files(lemma_dir, dec_files, "M", stopwords_set, pos_filter)
        target_f, _ = aggregate_counts_across_files(lemma_dir, dec_files, "F", stopwords_set, pos_filter)
        target_all = target_m + target_f
        target_total = sum(target_all.values())

        ref_m, _ = aggregate_counts_across_files(lemma_dir, other_files, "M", stopwords_set, pos_filter)
        ref_f, _ = aggregate_counts_across_files(lemma_dir, other_files, "F", stopwords_set, pos_filter)
        ref_all = ref_m + ref_f
        ref_total = sum(ref_all.values())

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_all, target_total, ref_all, ref_total, top_n)
        results["decade"] = dec
        return results

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_decade)(dec) for dec in decades
    )

    dec_results = [r for r in results if r is not None]
    if dec_results:
        dec_results_df = pd.concat(dec_results, ignore_index=True)
        dec_results_df = dec_results_df[["decade", "feature", "G2", "p"]]

        output_path = output_dir / "by_decade" / f"distinctive_{file_suffix}_by_decade_top{top_n}.csv"
        dec_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    # ══════════════════════════════════════════════════════════════════════
    # 3) By Decade and Category
    # ══════════════════════════════════════════════════════════════════════
    print("[3/6] Computing distinctive words by decade and category...")

    def process_decade_category(decade, cat):
        dc_files = metadata_df[
            (metadata_df["decade"] == decade) & (metadata_df["category"] == cat)
        ][["category", "decade"]].values.tolist()

        other_files = metadata_df[
            (metadata_df["decade"] != decade) | (metadata_df["category"] != cat)
        ][["category", "decade"]].values.tolist()

        if len(dc_files) == 0:
            return None

        target_m, _ = aggregate_counts_across_files(lemma_dir, dc_files, "M", stopwords_set, pos_filter)
        target_f, _ = aggregate_counts_across_files(lemma_dir, dc_files, "F", stopwords_set, pos_filter)
        target_all = target_m + target_f
        target_total = sum(target_all.values())

        ref_m, _ = aggregate_counts_across_files(lemma_dir, other_files, "M", stopwords_set, pos_filter)
        ref_f, _ = aggregate_counts_across_files(lemma_dir, other_files, "F", stopwords_set, pos_filter)
        ref_all = ref_m + ref_f
        ref_total = sum(ref_all.values())

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_all, target_total, ref_all, ref_total, top_n)
        results["decade"] = decade
        results["category"] = cat
        return results

    combos = [(d, c) for d in decades for c in categories]

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_decade_category)(d, c) for d, c in combos
    )

    dc_results = [r for r in results if r is not None]
    if dc_results:
        dc_results_df = pd.concat(dc_results, ignore_index=True)
        dc_results_df = dc_results_df[["decade", "category", "feature", "G2", "p"]]

        output_path = output_dir / "by_decade_and_category" / f"distinctive_{file_suffix}_by_decade_category_top{top_n}.csv"
        dc_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    # ══════════════════════════════════════════════════════════════════════
    # 4) By Gender and Category
    # ══════════════════════════════════════════════════════════════════════
    print("[4/6] Computing distinctive words by gender and category...")

    def process_gender_category(cat, gender):
        cat_files = metadata_df[metadata_df["category"] == cat][["category", "decade"]].values.tolist()
        if len(cat_files) == 0:
            return None

        other_gender = "F" if gender == "M" else "M"

        target_counts, target_total = aggregate_counts_across_files(lemma_dir, cat_files, gender, stopwords_set, pos_filter)
        ref_counts, ref_total = aggregate_counts_across_files(lemma_dir, cat_files, other_gender, stopwords_set, pos_filter)

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_counts, target_total, ref_counts, ref_total, top_n)
        results["category"] = cat
        results["gender"] = gender
        return results

    gc_combos = [(c, g) for c in categories for g in ["M", "F"]]

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_gender_category)(c, g) for c, g in gc_combos
    )

    gc_results = [r for r in results if r is not None]
    if gc_results:
        gc_results_df = pd.concat(gc_results, ignore_index=True)
        gc_results_df = gc_results_df[["category", "gender", "feature", "G2", "p"]]

        output_path = output_dir / "by_gender_and_category" / f"distinctive_{file_suffix}_by_gender_category_top{top_n}.csv"
        gc_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    # ══════════════════════════════════════════════════════════════════════
    # 5) By Gender and Decade
    # ══════════════════════════════════════════════════════════════════════
    print("[5/6] Computing distinctive words by gender and decade...")

    def process_gender_decade(dec, gender):
        dec_files = metadata_df[metadata_df["decade"] == dec][["category", "decade"]].values.tolist()
        if len(dec_files) == 0:
            return None

        other_gender = "F" if gender == "M" else "M"

        target_counts, target_total = aggregate_counts_across_files(lemma_dir, dec_files, gender, stopwords_set, pos_filter)
        ref_counts, ref_total = aggregate_counts_across_files(lemma_dir, dec_files, other_gender, stopwords_set, pos_filter)

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_counts, target_total, ref_counts, ref_total, top_n)
        results["decade"] = dec
        results["gender"] = gender
        return results

    gd_combos = [(d, g) for d in decades for g in ["M", "F"]]

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_gender_decade)(d, g) for d, g in gd_combos
    )

    gd_results = [r for r in results if r is not None]
    if gd_results:
        gd_results_df = pd.concat(gd_results, ignore_index=True)
        gd_results_df = gd_results_df[["decade", "gender", "feature", "G2", "p"]]

        output_path = output_dir / "by_gender_and_decade" / f"distinctive_{file_suffix}_by_gender_decade_top{top_n}.csv"
        gd_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    # ══════════════════════════════════════════════════════════════════════
    # 6) By Gender, Decade, and Category
    # ══════════════════════════════════════════════════════════════════════
    print("[6/6] Computing distinctive words by gender, decade, and category...")

    def process_gender_decade_category(cat, decade, gender):
        cache_df = load_lemma_cache(lemma_dir, cat, str(decade))
        if cache_df is None:
            return None

        other_gender = "F" if gender == "M" else "M"

        target_counts, target_total = get_word_counts_from_cache(cache_df, gender, stopwords_set, pos_filter)
        ref_counts, ref_total = get_word_counts_from_cache(cache_df, other_gender, stopwords_set, pos_filter)

        if target_total == 0 or ref_total == 0:
            return None

        results = compute_log_likelihood(target_counts, target_total, ref_counts, ref_total, top_n)
        results["category"] = cat
        results["decade"] = decade
        results["gender"] = gender
        return results

    gdc_combos = [(c, d, g) for c in categories for d in decades for g in ["M", "F"]]

    results = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(process_gender_decade_category)(c, d, g) for c, d, g in gdc_combos
    )

    gdc_results = [r for r in results if r is not None]
    if gdc_results:
        gdc_results_df = pd.concat(gdc_results, ignore_index=True)
        gdc_results_df = gdc_results_df[["category", "decade", "gender", "feature", "G2", "p"]]

        output_path = output_dir / "by_gender_decade_category" / f"distinctive_{file_suffix}_by_gender_decade_category_top{top_n}.csv"
        gdc_results_df.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path.name}\n")

    print("=" * 70)
    print(f"COMPLETE! All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
