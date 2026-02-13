# subset_by_gender.py

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd


def subset_by_gender(d: int | str, data_dir: str | Path) -> None:
    """
    Compatible with the new parse_congress.py + pipeline.

    Reads (produced by bind_data.py):
      - data/us_congress_spacy_parsed_<d>.parquet   (token-level; has speech_id, piece_id, sentence_id, token_id, token, pos, ...)
      - data/us_congress_spacy_ids_<d>.parquet      (speech-level; has speech_id, speech_order, gender, date, state, party, title, ...)

    Joins metadata onto token-level parsed data via speech_id, then writes:
      - data/gender_analysis/us_congress_men_<d>.parquet
      - data/gender_analysis/us_congress_women_<d>.parquet
    """
    d = str(d)
    data_dir = Path(data_dir)

    print(f"Processing {d}")

    parsed_file_name = f"us_congress_spacy_parsed_{d}.parquet"
    record_file_name = f"us_congress_spacy_ids_{d}.parquet"

    parsed_path = data_dir / parsed_file_name
    record_path = data_dir / record_file_name

    parsed_decade_subset = pd.read_parquet(parsed_path)
    record_decade_subset = pd.read_parquet(record_path)

    # --- Validate required columns ---
    parsed_required = ["speech_id", "piece_id", "sentence_id", "token_id", "token", "pos"]
    missing_parsed = [c for c in parsed_required if c not in parsed_decade_subset.columns]
    if missing_parsed:
        raise ValueError(f"Missing columns in {parsed_path}: {missing_parsed}")

    record_required = ["speech_id", "speech_order", "decade", "gender", "date", "state", "party", "title"]
    missing_record = [c for c in record_required if c not in record_decade_subset.columns]
    if missing_record:
        raise ValueError(f"Missing columns in {record_path}: {missing_record}")

    # Keep only what we need from the speech-level table (plus optional helpful provenance)
    keep_cols = record_required.copy()
    optional_cols = ["global_row_id", "chunk_id", "source_file", "doc_id"]
    for c in optional_cols:
        if c in record_decade_subset.columns:
            keep_cols.append(c)

    record_decade_subset = record_decade_subset[keep_cols].copy()

    # Avoid duplicate column names creating decade_x/decade_y
    if "decade" in parsed_decade_subset.columns and "decade" in record_decade_subset.columns:
        parsed_decade_subset = parsed_decade_subset.drop(columns=["decade"])

    joined_subset = parsed_decade_subset.merge(record_decade_subset, how="left", on="speech_id")

    del parsed_decade_subset, record_decade_subset
    gc.collect()

    # --- Output ---
    out_dir = Path("data") / "gender_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Columns to persist.
    # Must include: speech_id, speech_order, token, title, date (for downstream debate grouping)
    out_cols = [
        "speech_id",
        "speech_order",
        "piece_id",
        "sentence_id",
        "token_id",
        "token",
        "pos",
        "gender",
        "state",
        "party",
        "date",
        "title",
        "decade",
    ]

    # Keep optional provenance columns if present
    for c in ["global_row_id", "chunk_id", "source_file", "doc_id"]:
        if c in joined_subset.columns and c not in out_cols:
            out_cols.append(c)

    missing_out = [c for c in out_cols if c not in joined_subset.columns]
    if missing_out:
        raise ValueError(
            "Joined data is missing required columns for outputs: "
            + ", ".join(missing_out)
        )

    men = joined_subset.loc[joined_subset["gender"] == "M", out_cols].copy()
    men.to_parquet(out_dir / f"us_congress_men_{d}.parquet", index=False)

    women = joined_subset.loc[joined_subset["gender"] == "F", out_cols].copy()
    women.to_parquet(out_dir / f"us_congress_women_{d}.parquet", index=False)

    del men, women, joined_subset
    gc.collect()
