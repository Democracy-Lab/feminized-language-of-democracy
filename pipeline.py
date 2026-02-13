# pipeline.py

from __future__ import annotations

import os
import re
import platform
from pathlib import Path
from typing import Sequence

import pandas as pd

from create_decade_subset import create_decades_col, split_by_decade
import bind_data
import parse_congress
from subset_by_gender import subset_by_gender
from remove_stopwords import load_congress_stopwords, remove_stopwords
from subset_by_category import subset_by_category


DEBUG = False
SHOULD_DELETE_CHUNKS = False

DECADES: Sequence[int] = (
    1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940,
    1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020
)

DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
GENDER_DIR = DATA_DIR / "gender_analysis"
CATEGORIES_DIR = GENDER_DIR / "categories"

INPUT_CSV = "/local/scratch/group/guldigroup/climate_change/congress/OCR_testing/combined_congressional_records_12-07-25.csv"

NUM_CHUNKS = 384

MAX_CHARS = 900_000
SPLIT_THRESHOLD_CHARS = 500_000

SKIP_IF_DECADE_PARQUETS_EXIST = True
SKIP_PARSE_IF_PARSED_DECADE_EXISTS = True
BIND_ONLY_IF_OUTPUT_MISSING = True

def ensure_dirs() -> None:
    for p, msg in [
        (DATA_DIR, "data directory"),
        (CHUNKS_DIR, "chunks directory"),
        (GENDER_DIR, "gender analysis directory"),
        (CATEGORIES_DIR, "categories directory in gender analysis directory"),
    ]:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"Created {msg}")
        else:
            print(f"{msg.capitalize()} already exists")


def infer_workers(os_name: str) -> int:
    if os_name == "Windows":
        return 20
    if os_name == "Linux":
        return 34
    return max(1, (os.cpu_count() or 4) - 1)


def decade_from_filename(path: Path) -> str:
    m = re.search(r"\d{4}", path.name)
    if not m:
        raise ValueError(f"Could not extract decade from filename: {path}")
    return m.group(0)


def main() -> None:
    ensure_dirs()

    # ----------------------------
    # 1) Ensure decade parquet inputs exist (or create them)
    # ----------------------------
    file_list = sorted(DATA_DIR.glob("us_congress_????.parquet"))

    if SKIP_IF_DECADE_PARQUETS_EXIST and len(file_list) == len(DECADES):
        print("Found decade parquet files already; skipping CSV load + split_by_decade.")
    else:
        print(f"Loading CSV: {INPUT_CSV}")
        us_congress_data = pd.read_csv(INPUT_CSV)

        us_congress_data = create_decades_col(us_congress_data)

        if DEBUG:
            us_congress_data = (
                us_congress_data
                .groupby("decade", group_keys=False)
                .apply(lambda g: g.sample(n=min(100, len(g)), random_state=0))
                .reset_index(drop=True)
            )

        split_by_decade(us_congress_data, DATA_DIR)
        file_list = sorted(DATA_DIR.glob("us_congress_????.parquet"))

    operating_system = platform.system()
    workers = infer_workers(operating_system)

    print(f"Operating system: {operating_system}")
    print(f"Workers (cores): {workers}")

    # ----------------------------
    # 2) Parse only what’s needed
    #    - If decade-level parsed parquet exists, skip parsing entirely for that decade.
    #    - Otherwise, parse_congress.process() will only parse missing chunk outputs.
    # ----------------------------
    existing_parsed_decades: set[str] = set()
    if SKIP_PARSE_IF_PARSED_DECADE_EXISTS:
        for d in DECADES:
            out = DATA_DIR / f"us_congress_spacy_parsed_{d}.parquet"
            if out.exists():
                existing_parsed_decades.add(str(d))

    for f in file_list:
        current_decade = decade_from_filename(f)

        if SKIP_PARSE_IF_PARSED_DECADE_EXISTS and current_decade in existing_parsed_decades:
            print(f"[{current_decade}] Found decade-level spacy parsed parquet; skipping parse.")
            continue

        print(f"Loading: {f}")
        decade_subset = pd.read_parquet(f)

        if hasattr(parse_congress, "process"):
            parse_congress.process(
                decade_subset,
                current_decade,
                num_chunks=NUM_CHUNKS,
                operating_system=operating_system,
                max_chars=MAX_CHARS,
                split_threshold_chars=SPLIT_THRESHOLD_CHARS,
                workers=workers,
            )
        else:
            raise AttributeError(
                "parse_congress.py must define process(decade_subset, decade, num_chunks, operating_system, ...)"
            )

    # ----------------------------
    # 3) Bind only missing decade-level outputs (fast resume)
    # ----------------------------
    def _missing_outputs(prefix: str) -> list[int]:
        missing: list[int] = []
        for d in DECADES:
            if not (DATA_DIR / f"{prefix}{d}.parquet").exists():
                missing.append(int(d))
        return missing

    # spacy parsed decade-level files
    if BIND_ONLY_IF_OUTPUT_MISSING:
        missing = _missing_outputs("us_congress_spacy_parsed_")
        if missing:
            bind_data.bind_chunks(
                chunk_type="spacy_parse",
                decades=missing,
                chunk_dir=CHUNKS_DIR,
                output_dir=DATA_DIR,
            )
        else:
            print("All us_congress_spacy_parsed_<decade>.parquet files exist; skipping bind (spacy_parse).")
    else:
        bind_data.bind_chunks(chunk_type="spacy_parse", decades=DECADES, chunk_dir=CHUNKS_DIR, output_dir=DATA_DIR)

    # base/ids decade-level files
    if BIND_ONLY_IF_OUTPUT_MISSING:
        missing = _missing_outputs("us_congress_spacy_ids_")
        if missing:
            bind_data.bind_chunks(
                chunk_type="default",
                decades=missing,
                chunk_dir=CHUNKS_DIR,
                output_dir=DATA_DIR,
            )
        else:
            print("All us_congress_spacy_ids_<decade>.parquet files exist; skipping bind (default).")
    else:
        bind_data.bind_chunks(chunk_type="default", decades=DECADES, chunk_dir=CHUNKS_DIR, output_dir=DATA_DIR)

    # ----------------------------
    # 4) Subset by gender (resume-safe)
    # ----------------------------
    for d in DECADES:
        d_str = str(d)

        parsed_decade = DATA_DIR / f"us_congress_spacy_parsed_{d_str}.parquet"
        ids_decade = DATA_DIR / f"us_congress_spacy_ids_{d_str}.parquet"

        if not parsed_decade.exists() or not ids_decade.exists():
            print(f"[{d_str}] Missing parsed/ids decade parquet(s); skipping subset_by_gender.")
            continue

        men_out = GENDER_DIR / f"us_congress_men_{d_str}.parquet"
        women_out = GENDER_DIR / f"us_congress_women_{d_str}.parquet"

        if men_out.exists() and women_out.exists():
            print(f"[{d_str}] Men/Women outputs exist; skipping subset_by_gender.")
        else:
            subset_by_gender(d_str, DATA_DIR)

    # ----------------------------
    # 5) Remove stopwords + create men_and_women_clean (resume-safe)
    # ----------------------------
    stopwords_set = load_congress_stopwords()

    for d in DECADES:
        d_str = str(d)

        men_path = GENDER_DIR / f"us_congress_men_{d_str}.parquet"
        women_path = GENDER_DIR / f"us_congress_women_{d_str}.parquet"
        clean_out = GENDER_DIR / f"us_congress_men_and_women_clean_{d_str}.parquet"

        if clean_out.exists():
            print(f"[{d_str}] Clean men+women file exists; skipping stopword removal.")
            continue

        if not men_path.exists() or not women_path.exists():
            print(f"[{d_str}] Missing men/women parquet(s); skipping stopword removal.")
            continue

        print(f"[{d_str}] Removing stopwords + writing clean men+women parquet...")

        parsed_men = pd.read_parquet(men_path)
        parsed_women = pd.read_parquet(women_path)

        out1 = remove_stopwords(parsed_men, stopwords_set)
        out2 = remove_stopwords(parsed_women, stopwords_set)

        all_data = pd.concat([out1, out2], ignore_index=True)
        all_data.to_parquet(clean_out, index=False)

        del parsed_men, parsed_women, out1, out2, all_data

    # ----------------------------
    # 6) Categorization (resume-safe)
    # ----------------------------
    for d in DECADES:
        d_str = str(d)

        clean_path = GENDER_DIR / f"us_congress_men_and_women_clean_{d_str}.parquet"
        if not clean_path.exists():
            print(f"[{d_str}] Missing clean men+women parquet; skipping categorization.")
            continue

        # Sentinel output (default thr/pom): debate_map_<decade>.csv in tfidf_norm_qc_thr0.030_pom0.85
        qc_dir = CATEGORIES_DIR / f"tfidf_norm_qc_thr{0.03:.3f}_pom{0.85:.2f}"
        sentinel = qc_dir / f"debate_map_{d_str}.csv"

        if sentinel.exists():
            print(f"[{d_str}] Categorization outputs appear to exist; skipping categorization.")
            continue

        print(f"[{d_str}] Running categorization...")
        f = pd.read_parquet(clean_path)
        subset_by_category(f, d_str, CATEGORIES_DIR)
        del f

    if SHOULD_DELETE_CHUNKS:
        bind_data.delete_chunks(chunk_dir=CHUNKS_DIR)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()