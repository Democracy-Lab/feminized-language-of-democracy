# bind_data.py
from __future__ import annotations

import gc
import re
import shutil
import time
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = Path("data")
CHUNK_DIR = DATA_DIR / "chunks"

# ----------------------------
# Helpers
# ----------------------------
_CHUNK_ID_RE = re.compile(r"_chunk_(\d+)\.parquet$", re.I)


def _chunk_sort_key(p: Path) -> tuple[int, str]:
    """
    Sort by numeric chunk id first (so chunk_2 comes before chunk_10),
    then by filename as a stable tiebreaker.
    """
    m = _CHUNK_ID_RE.search(p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (10**18, p.name)


def delete_chunks(chunk_dir: Path = CHUNK_DIR) -> None:
    """
    Prompts for confirmation. If confirmed, deletes everything inside data/chunks.
    """
    answer = input(
        "Are you sure you want to delete all chunks? This action cannot be reversed. [y/N]: "
    ).strip()

    if answer.lower() != "y":
        print("Operation cancelled.")
        return

    if not chunk_dir.exists():
        print(f"Chunk directory does not exist: {chunk_dir}")
        return

    files_deleted = 0
    for p in chunk_dir.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            files_deleted += 1
        except Exception:
            print(f"Skipping (delete error): {p}")

    print(f"Deleted {files_deleted} file(s) from: {chunk_dir}")


def _read_table_with_chunk_file(file_path: Path) -> pa.Table | None:
    """
    Read a parquet file as a PyArrow Table and add chunk_file column.
    Returns None on error (mirrors your safe_load_df behavior).
    """
    try:
        table = pq.read_table(file_path)

        # Add provenance column exactly like prior code
        chunk_file_arr = pa.array([file_path.name] * table.num_rows, type=pa.string())
        table = table.append_column("chunk_file", chunk_file_arr)

        return table
    except Exception:
        print(f"Skipping file due to error: {file_path}")
        return None

    
def _union_schema(paths: list[Path]) -> pa.schema:
    """
    Build a union schema across all parquet files in `paths`.
    This approximates pandas concat's behavior (union columns).
    We take the first-seen type for a column; later type mismatches are cast when possible.
    """
    fields: dict[str, pa.Field] = {}

    for p in paths:
        try:
            pf = pq.ParquetFile(p)
            sch = pf.schema_arrow
        except Exception:
            # keep behavior: bad file will be skipped later anyway
            continue

        for f in sch:
            if f.name not in fields:
                fields[f.name] = f
            else:
                prev = fields[f.name]
                # FIX: upgrade null-typed columns if a later chunk has a real type
                if pa.types.is_null(prev.type) and not pa.types.is_null(f.type):
                    fields[f.name] = pa.field(f.name, f.type, nullable=True)

    # chunk_file is always added
    if "chunk_file" not in fields:
        fields["chunk_file"] = pa.field("chunk_file", pa.string())

    # Deterministic field ordering
    names = sorted([n for n in fields.keys() if n != "chunk_file"])
    names.append("chunk_file")

    return pa.schema([fields[n] for n in names])



def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """
    Ensure `table` has exactly the columns in `schema`:
      - add missing cols as nulls
      - reorder cols to match schema
      - cast cols to schema types when possible
    """
    cols = {name: table.column(name) for name in table.column_names}

    arrays: list[pa.Array | pa.ChunkedArray] = []
    for field in schema:
        name = field.name
        if name in cols:
            col = cols[name]
            # Cast if needed/possible
            if not col.type.equals(field.type):
                try:
                    col = col.cast(field.type)
                except Exception:
                    # If cast fails, keep original type (still writes, but may differ from union schema)
                    # In your pipeline, chunk schemas should be consistent, so this shouldn't happen.
                    pass
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))

    return pa.Table.from_arrays(arrays, schema=schema)


def bind_chunks(
    chunk_type: str,
    decades: Sequence[int | str],
    chunk_dir: Path = CHUNK_DIR,
    output_dir: Path = DATA_DIR,
) -> None:
    """
    Binds per-chunk parquet files into one parquet per decade.

    Output is equivalent to prior version:
      - same chunk selection
      - same deterministic chunk ordering
      - same concatenation order (chunk_1, chunk_2, ...)
      - same added provenance column: chunk_file
      - same output filenames/prefixes

    Key upgrade:
      - stream write to parquet (bounded memory)
    """
    if chunk_type == "spacy_parse":
        match_pattern = re.compile(r"^us_congress_spacy_parsed_\d{4}_chunk_\d+\.parquet$", re.I)
        detection_pattern = r"us_congress_spacy_parsed_%s_chunk_\d+\.parquet$"
        output_prefix = "us_congress_spacy_parsed_"

    elif chunk_type == "default":
        match_pattern = re.compile(r"^us_congress_\d{4}_chunk_\d+\.parquet$", re.I)
        detection_pattern = r"us_congress_%s_chunk_\d+\.parquet$"
        output_prefix = "us_congress_spacy_ids_"
    else:
        raise ValueError('chunk_type must be one of: "spacy_parse", "default"')

    if not chunk_dir.exists():
        print(f"Chunk directory does not exist: {chunk_dir}")
        return

    all_chunk_files = sorted(
        [p for p in chunk_dir.iterdir() if p.is_file() and match_pattern.match(p.name)],
        key=_chunk_sort_key,
    )

    files_by_decade: dict[str, list[Path]] = {}
    for d in decades:
        d_str = str(d)
        decade_re = re.compile(detection_pattern % re.escape(d_str), re.I)
        files_by_decade[d_str] = [p for p in all_chunk_files if decade_re.search(p.name)]

    output_dir.mkdir(parents=True, exist_ok=True)

    for d_str, decade_files in files_by_decade.items():
        print(f"Processing decade: {d_str}")
        t0 = time.perf_counter()

        if not decade_files:
            print(f"No files found for decade {d_str}")
            continue

        decade_files = sorted(decade_files, key=_chunk_sort_key)

        # Build a union schema up front (cheap) so we can stream-write consistently.
        target_schema = _union_schema(decade_files)

        out_path = output_dir / f"{output_prefix}{d_str}.parquet"
        writer: pq.ParquetWriter | None = None

        rows_written = 0
        files_used = 0

        for fp in decade_files:
            table = _read_table_with_chunk_file(fp)
            if table is None:
                continue

            table = _align_table_to_schema(table, target_schema)

            if writer is None:
                writer = pq.ParquetWriter(out_path, target_schema, compression="snappy")

            writer.write_table(table)
            rows_written += table.num_rows
            files_used += 1

            # help Python release references sooner
            del table

        if writer is not None:
            writer.close()

        dt = time.perf_counter() - t0
        if files_used == 0:
            print(f"All files failed to load for decade {d_str}; skipping output.")
            # if an empty file was created, remove it
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
        else:
            print(f"Wrote {rows_written:,} rows from {files_used} chunk(s) in {dt:.2f}s -> {out_path.name}")

        gc.collect()
