from __future__ import annotations

import re
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import spacy


# ----------------------------
# CONFIG YOU MUST SET
# ----------------------------
# Equivalent of your R `chunks_dir`
CHUNKS_DIR = Path("PATH/TO/chunks_dir")  # <-- set me


# ----------------------------
# Helpers: filename -> ids
# ----------------------------
_CHUNK_ID_RE = re.compile(r"_chunk_(\d+)\.parquet$", re.IGNORECASE)

def _extract_chunk_ids(files: Iterable[str]) -> List[int]:
    ids: List[int] = []
    for name in files:
        m = _CHUNK_ID_RE.search(name)
        if m:
            try:
                ids.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(set(ids))


def existing_chunk_ids(decade: str, chunks_dir: Path = CHUNKS_DIR) -> List[int]:
    """
    Match: ^us_congress_{decade}_chunk_\\d+\\.parquet$
    Return sorted unique chunk IDs (ints).
    """
    pat = re.compile(rf"^us_congress_{re.escape(str(decade))}_chunk_\d+\.parquet$", re.IGNORECASE)
    files = [p.name for p in chunks_dir.iterdir() if p.is_file() and pat.match(p.name)]
    return _extract_chunk_ids(files)


def existing_parsed_ids(decade: str, chunks_dir: Path = CHUNKS_DIR) -> List[int]:
    """
    Match: ^us_congress_spacy_parsed_{decade}_chunk_\\d+\\.parquet$
    Return sorted unique chunk IDs (ints).
    """
    pat = re.compile(
        rf"^us_congress_spacy_parsed_{re.escape(str(decade))}_chunk_\d+\.parquet$",
        re.IGNORECASE,
    )
    files = [p.name for p in chunks_dir.iterdir() if p.is_file() and pat.match(p.name)]
    return _extract_chunk_ids(files)


# ----------------------------
# Chunking: dplyr::ntile(row_number(), num_chunks)
# ----------------------------
def _ntile_slices(n_rows: int, num_chunks: int) -> List[tuple[int, int]]:
    """
    Replicates ntile(row_number(), num_chunks) on the current row order:
      - chunk sizes differ by at most 1
      - earlier chunks get the extra rows
    Returns 0-based [start, end) slices for each chunk_id = 1..num_chunks
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if n_rows < 0:
        raise ValueError("n_rows must be >= 0")

    q, r = divmod(n_rows, num_chunks)  # q = base size, r = first r chunks get +1
    slices: List[tuple[int, int]] = []
    start = 0
    for i in range(1, num_chunks + 1):
        size = q + 1 if i <= r else q
        end = start + size
        slices.append((start, end))
        start = end
    return slices


@dataclass(frozen=True)
class ChunkView:
    """
    Like your R `chunks <- make_chunks(decade_df, num_chunks)`, but without copying
    every chunk into a list. Access with 1-based chunk_id.
    """
    df: pd.DataFrame
    slices: List[tuple[int, int]]

    @property
    def num_chunks(self) -> int:
        return len(self.slices)

    def get(self, chunk_id: int) -> pd.DataFrame:
        # R list indexing is 1-based; your R code uses chunks[[chunk_id]]
        if not (1 <= chunk_id <= self.num_chunks):
            raise IndexError(f"chunk_id {chunk_id} out of range 1..{self.num_chunks}")
        start, end = self.slices[chunk_id - 1]
        # copy() to avoid accidental in-place edits leaking into the master df
        return self.df.iloc[start:end].copy()


def make_chunks(decade_df: pd.DataFrame, num_chunks: int) -> ChunkView:
    slices = _ntile_slices(len(decade_df), num_chunks)
    return ChunkView(df=decade_df, slices=slices)


# ----------------------------
# spaCy init + parse hooks
# ----------------------------
_NLP = None

def spacy_initialize(model: str = "en_core_web_sm", refresh_settings: bool = True):
    """
    Rough Python equivalent of spacyr::spacy_initialize().
    `refresh_settings` doesn't have a direct meaning in spaCy-Python, so we ignore it.
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(model)
    return _NLP


def spacy_parse_windows(df_chunk: pd.DataFrame, chunk_id: int, decade: str, num_chunks: int):
    """
    TODO: implement based on your R spacy_parse_windows().
    Should write outputs with the same naming scheme used by existing_chunk_ids/existing_parsed_ids.
    """
    raise NotImplementedError("Need your R spacy_parse_windows() behavior to replicate.")


def spacy_parse_unix(df_chunk: pd.DataFrame, chunk_id: int, decade: str, num_chunks: int):
    """
    TODO: implement based on your R spacy_parse_unix().
    """
    raise NotImplementedError("Need your R spacy_parse_unix() behavior to replicate.")


def run_chunk_index(chunks: ChunkView, decade: str, chunk_id: int, os_name: str):
    print(f"Backfilling chunk_id {chunk_id}")

    df_chunk = chunks.get(chunk_id)

    spacy_initialize(model="en_core_web_sm", refresh_settings=True)

    if os_name == "Windows":
        spacy_parse_windows(df_chunk, chunk_id, decade, num_chunks=chunks.num_chunks)
    else:
        spacy_parse_unix(df_chunk, chunk_id, decade, num_chunks=chunks.num_chunks)


# ----------------------------
# Main backfill function
# ----------------------------
def backfill_missing_chunks(
    decade_file_paths: Sequence[str | Path],
    decades: Optional[Sequence[str] | str] = None,
    num_chunks: int = 800,
    os_name: Optional[str] = None,
    print_existing: bool = True,
    print_missing_parsed_too: bool = True,
    chunks_dir: Path = CHUNKS_DIR,
) -> bool:
    # normalize inputs
    decade_paths = [Path(p) for p in decade_file_paths]

    if os_name is None:
        os_name = platform.system()  # "Windows", "Linux", "Darwin", etc.

    if decades is None:
        # mimic: str_extract(basename(decade_file_paths), "\\d{4}")
        found = []
        for p in decade_paths:
            m = re.search(r"\d{4}", p.name)
            if m:
                found.append(m.group(0))
        decades_list = sorted(set(found))
    elif isinstance(decades, str):
        decades_list = [decades]
    else:
        decades_list = list(decades)

    for decade in decades_list:
        print(f"Decade: {decade}")

        # Identify decade parquet uniquely:
        # ^us_congress_{decade}\\.parquet$
        pat = re.compile(rf"^us_congress_{re.escape(str(decade))}\.parquet$", re.IGNORECASE)
        matches = [p for p in decade_paths if pat.match(p.name)]

        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely identify decade parquet for {decade}. Matches: {matches}")

        f = matches[0]

        have_chunks = existing_chunk_ids(decade, chunks_dir=chunks_dir)
        target_ids = list(range(1, num_chunks + 1))
        missing_base = sorted(set(target_ids) - set(have_chunks))

        if print_existing:
            print(f"Existing base chunk_ids ({len(have_chunks)}):")
            print(have_chunks)

        print(f"\nBackfilling BASE chunk_ids ({len(missing_base)}):")
        if missing_base:
            print(missing_base)
        else:
            print("None 🎉")

        missing_parsed: List[int] = []
        if print_missing_parsed_too:
            have_parsed = existing_parsed_ids(decade, chunks_dir=chunks_dir)
            missing_parsed = sorted(set(target_ids) - set(have_parsed))
            print(f"\nMissing PARSED chunk_ids ({len(missing_parsed)}):")
            if missing_parsed:
                print(missing_parsed)
            else:
                print("None")

        if len(missing_base) == 0 and (not print_missing_parsed_too or len(missing_parsed) == 0):
            continue

        print(f"\nLoading decade data from:\n{f}")
        decade_df = pd.read_parquet(f)

        print(f"Rebuilding chunk view (num_chunks = {num_chunks})")
        chunks = make_chunks(decade_df, num_chunks)

        if missing_base:
            print("\nExecuting backfill for BASE chunk_ids:")
            print(missing_base)
            for cid in missing_base:
                run_chunk_index(chunks, decade, cid, os_name)

        if print_missing_parsed_too:
            have_parsed_after = existing_parsed_ids(decade, chunks_dir=chunks_dir)
            missing_parsed_after = sorted(set(target_ids) - set(have_parsed_after))

            if missing_parsed_after:
                print("\nExecuting backfill for PARSED-only chunk_ids:")
                print(missing_parsed_after)
                for cid in missing_parsed_after:
                    run_chunk_index(chunks, decade, cid, os_name)
            else:
                print("All parsed chunks exist after base backfill")

        print(f"Done decade {decade}")

    return True