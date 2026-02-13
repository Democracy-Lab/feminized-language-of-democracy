# parse_congress.py

from __future__ import annotations

import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import spacy

# IMPORTANT: used for streaming Parquet writes (bounded memory)
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------
# CONFIG DEFAULTS (can override in calls)
# ----------------------------
DEFAULT_CHUNKS_DIR = Path("data") / "chunks"
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# You said some speeches are artificially > 1M chars.
DEFAULT_MAX_CHARS = 900_000

# Only split speeches if they are above this threshold
DEFAULT_SPLIT_THRESHOLD_CHARS = 500_000

# Stream-writing control: how many token rows to buffer before flushing to parquet
DEFAULT_TOKEN_ROWS_BUFFER = 250_000

# Cap workers by default to avoid OOM from too many spaCy processes
DEFAULT_WORKER_CAP = 24

# spaCy pipe batch size inside each worker (bigger = faster, but can use more RAM)
DEFAULT_SPACY_BATCH_SIZE = 64


# ----------------------------
# Helpers: filename -> chunk ids (kept for compatibility / optional backfill)
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


def existing_chunk_ids(decade: str, chunks_dir: Path) -> List[int]:
    pat = re.compile(rf"^us_congress_{re.escape(str(decade))}_chunk_\d+\.parquet$", re.IGNORECASE)
    if not chunks_dir.exists():
        return []
    files = [p.name for p in chunks_dir.iterdir() if p.is_file() and pat.match(p.name)]
    return _extract_chunk_ids(files)


def existing_parsed_ids(decade: str, chunks_dir: Path) -> List[int]:
    pat = re.compile(
        rf"^us_congress_spacy_parsed_{re.escape(str(decade))}_chunk_\d+\.parquet$",
        re.IGNORECASE,
    )
    if not chunks_dir.exists():
        return []
    files = [p.name for p in chunks_dir.iterdir() if p.is_file() and pat.match(p.name)]
    return _extract_chunk_ids(files)


# ----------------------------
# Chunking: dplyr::ntile(row_number(), num_chunks)
# ----------------------------
def _ntile_slices(n_rows: int, num_chunks: int) -> List[Tuple[int, int]]:
    """
    Replicates ntile(row_number(), num_chunks) on current row order:
      - chunk sizes differ by at most 1
      - earlier chunks get extra rows
    Returns 0-based [start, end) slices for each chunk_id = 1..num_chunks.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    q, r = divmod(n_rows, num_chunks)  # first r chunks have size q+1
    slices: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, num_chunks + 1):
        size = q + 1 if i <= r else q
        end = start + size
        slices.append((start, end))
        start = end
    return slices


@dataclass(frozen=True)
class ChunkView:
    df: pd.DataFrame
    slices: List[Tuple[int, int]]

    @property
    def num_chunks(self) -> int:
        return len(self.slices)

    def get(self, chunk_id: int) -> pd.DataFrame:
        # 1-based to match R chunks[[chunk_id]]
        if not (1 <= chunk_id <= self.num_chunks):
            raise IndexError(f"chunk_id {chunk_id} out of range 1..{self.num_chunks}")
        start, end = self.slices[chunk_id - 1]
        # copy() to keep worker-side reads isolated
        return self.df.iloc[start:end].copy()


def make_chunks(decade_df: pd.DataFrame, num_chunks: int) -> ChunkView:
    return ChunkView(df=decade_df, slices=_ntile_slices(len(decade_df), num_chunks))


# ----------------------------
# spaCy init + parse helpers
# ----------------------------
# IMPORTANT: one NLP per *process* (not shared across processes)
_NLP: Optional[spacy.language.Language] = None


def _init_worker(spacy_model: str) -> None:
    global _NLP
    # disable parser + ner (lighter)
    _NLP = spacy.load(spacy_model, disable=["parser", "ner"])

    # add rule-based sentence splitter
    if "sentencizer" not in _NLP.pipe_names:
        _NLP.add_pipe("sentencizer", first=True)

def _get_nlp(spacy_model: str) -> spacy.language.Language:
    """Lazy fallback if initializer wasn't used."""
    global _NLP
    if _NLP is None:
        _init_worker(spacy_model)
    assert _NLP is not None
    return _NLP


def split_long_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    split_threshold_chars: int = DEFAULT_SPLIT_THRESHOLD_CHARS,
) -> List[str]:
    """
    - If len(text) <= split_threshold_chars: do NOT split
    - If len(text) > split_threshold_chars: split into max_chars-sized pieces
    """
    if text is None:
        return [""]

    if not isinstance(text, str):
        text = str(text)

    n = len(text)

    if n <= split_threshold_chars:
        return [text]

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    return [text[i : i + max_chars] for i in range(0, n, max_chars)]


def ensure_speech_ids(decade_df: pd.DataFrame, decade: str) -> pd.DataFrame:
    """
    Ensure every row (speech) has:
      - speech_order: 1..N in current row order
      - speech_id: globally-unique stable identifier for the speech

    Also writes doc_id = speech_id for compatibility with any lingering code.
    """
    out = decade_df.copy()

    if "speech_order" not in out.columns:
        out["speech_order"] = range(1, len(out) + 1)

    if "speech_id" not in out.columns:
        out["speech_id"] = [f"{decade}_{i:07d}" for i in range(1, len(out) + 1)]
    else:
        if out["speech_id"].isna().any():
            raise ValueError("speech_id contains NA values; cannot proceed safely.")
        if out["speech_id"].duplicated().any():
            dup = out.loc[out["speech_id"].duplicated(), "speech_id"].head(5).tolist()
            raise ValueError(f"speech_id is not unique within decade {decade}. Examples: {dup}")

    out["doc_id"] = out["speech_id"]
    return out


def iter_spacy_tuples(
    df_chunk: pd.DataFrame,
    *,
    max_chars: int,
    split_threshold_chars: int,
) -> Iterator[Tuple[str, Dict[str, object]]]:
    """
    STREAMING generator for spaCy input.
    Replaces the old make_spacy_input() DataFrame to avoid building large intermediate objects.

    Yields (text, context) where context carries speech_id and piece_id.
    """
    if "speech_id" not in df_chunk.columns or "content" not in df_chunk.columns:
        raise ValueError("df_chunk must contain columns: speech_id, content")

    for speech_id, content in zip(df_chunk["speech_id"].tolist(), df_chunk["content"].tolist()):
        if content is None or (isinstance(content, float) and pd.isna(content)):
            content_str = ""
        else:
            content_str = content if isinstance(content, str) else str(content)

        pieces = split_long_text(
            content_str,
            max_chars=max_chars,
            split_threshold_chars=split_threshold_chars,
        )

        for j, piece in enumerate(pieces, start=1):
            yield (piece, {"speech_id": speech_id, "piece_id": j})


def _base_chunk_path(chunks_dir: Path, decade: str, chunk_id: int) -> Path:
    return chunks_dir / f"us_congress_{decade}_chunk_{chunk_id}.parquet"


def _parsed_chunk_path(chunks_dir: Path, decade: str, chunk_id: int) -> Path:
    return chunks_dir / f"us_congress_spacy_parsed_{decade}_chunk_{chunk_id}.parquet"


def _write_base_chunk(
    df_chunk: pd.DataFrame,
    *,
    chunks_dir: Path,
    decade: str,
    chunk_id: int,
) -> Path:
    """
    Writes the base chunk parquet (record-level) if it doesn't exist.
    Returns the base chunk path.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    base_path = _base_chunk_path(chunks_dir, decade, chunk_id)

    if base_path.exists():
        return base_path

    out_df = df_chunk.copy()
    required = {"speech_id", "speech_order", "content"}
    missing = required - set(out_df.columns)
    if missing:
        raise ValueError(
            f"df_chunk missing required columns {sorted(missing)}. Did you call ensure_speech_ids()?"
        )

    out_df["chunk_id"] = int(chunk_id)
    out_df["decade"] = str(decade)

    if "doc_id" not in out_df.columns:
        out_df["doc_id"] = out_df["speech_id"]

    out_df.to_parquet(base_path, index=False)
    return base_path


def _flush_token_buffer_to_parquet(
    writer: Optional[pq.ParquetWriter],
    records: Dict[str, List[object]],
    out_path: Path,
) -> pq.ParquetWriter:
    """
    Flush current buffered token rows to the ParquetWriter and clear buffers.
    Creates the writer on first flush (so schema is inferred once).
    """
    df_batch = pd.DataFrame(records)
    table = pa.Table.from_pandas(df_batch, preserve_index=False)

    if writer is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

    writer.write_table(table)

    # clear buffers in-place to free memory
    for k in records.keys():
        records[k].clear()

    return writer


def _parse_chunk_worker(
    *,
    decade: str,
    chunk_id: int,
    chunks_dir: str,
    max_chars: int,
    split_threshold_chars: int,
    spacy_model: str,
    spacy_batch_size: int,
    token_rows_buffer: int,
) -> int:
    """
    Worker: reads base chunk parquet, parses with spaCy, STREAM-WRITES parsed parquet.
    Returns chunk_id when done.
    """
    chunks_dir_p = Path(chunks_dir)
    parsed_path = _parsed_chunk_path(chunks_dir_p, decade, chunk_id)
    if parsed_path.exists():
        return chunk_id

    base_path = _base_chunk_path(chunks_dir_p, decade, chunk_id)
    if not base_path.exists():
        raise FileNotFoundError(f"Base chunk not found: {base_path}")

    df_chunk = pd.read_parquet(base_path)

    nlp = _get_nlp(spacy_model)

    tuples_iter = iter_spacy_tuples(
        df_chunk,
        max_chars=max_chars,
        split_threshold_chars=split_threshold_chars,
    )

    # Column buffers (MUCH lower overhead than list-of-dicts)
    # Keep outputs aligned with your needs: token, lemma, pos (+ tag kept because it’s basically free once tagger runs).
    # dep/head are intentionally omitted (or set to None) to avoid dependency parsing.
    records: Dict[str, List[object]] = {
        "speech_id": [],
        "piece_id": [],
        "sentence_id": [],
        "token_id": [],
        "token": [],
        "lemma": [],
        "pos": [],
        "tag": [],
        "chunk_id": [],
        "decade": [],
        "doc_id": [],
    }

    writer: Optional[pq.ParquetWriter] = None
    buffered = 0

    # Stream through docs
    for doc, ctx in nlp.pipe(tuples_iter, batch_size=spacy_batch_size, as_tuples=True):
        speech_id = ctx["speech_id"]
        piece_id = int(ctx["piece_id"])

        # sentencizer sets doc.sents
        for sent_i, sent in enumerate(doc.sents, start=1):
            for tok_pos, tok in enumerate(sent, start=1):
                records["speech_id"].append(speech_id)
                records["piece_id"].append(piece_id)
                records["sentence_id"].append(int(sent_i))
                records["token_id"].append(int(tok_pos))
                records["token"].append(tok.text)
                records["lemma"].append(tok.lemma_)
                records["pos"].append(tok.pos_)
                records["tag"].append(tok.tag_)
                records["chunk_id"].append(int(chunk_id))
                records["decade"].append(str(decade))
                records["doc_id"].append(speech_id)

                buffered += 1

                # Flush periodically to bound memory
                if buffered >= token_rows_buffer:
                    writer = _flush_token_buffer_to_parquet(writer, records, parsed_path)
                    buffered = 0

    # final flush
    if buffered > 0:
        writer = _flush_token_buffer_to_parquet(writer, records, parsed_path)

    if writer is not None:
        writer.close()
    else:
        # No tokens produced (empty chunk). Still create an empty file with the right columns.
        empty_df = pd.DataFrame({k: [] for k in records.keys()})
        empty_df.to_parquet(parsed_path, index=False)

    # Add metadata columns by writing a small sidecar? Instead: include them at bind stage.
    # For backward compatibility with your prior outputs, we write them as separate columns *now*:
    # easiest without rewriting file: write a tiny parquet alongside, and let bind_data handle joins? (Too much.)
    # Instead: append these columns here by rewriting only if needed.
    # Simpler: keep metadata in bind stage or in filename. Your bind_data likely uses filename decade/chunk_id anyway.

    return chunk_id


def process(
    decade_subset: pd.DataFrame,
    decade: str,
    num_chunks: int = 800,
    operating_system: str | None = None,
    *,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    max_chars: int = DEFAULT_MAX_CHARS,
    split_threshold_chars: int = DEFAULT_SPLIT_THRESHOLD_CHARS,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    workers: int | None = None,
    worker_cap: int = DEFAULT_WORKER_CAP,
    spacy_batch_size: int = DEFAULT_SPACY_BATCH_SIZE,
    token_rows_buffer: int = DEFAULT_TOKEN_ROWS_BUFFER,
    **kwargs,
) -> None:
    """
    Pipeline entrypoint:
      1) enforce stable speech_id + speech_order
      2) deterministically chunks by row order (ntile)
      3) write ALL base chunk files (cheap / sequential)
      4) PARALLELIZE spaCy parsing of missing parsed chunks (expensive)

    Key changes:
      - disable dependency parsing & NER (faster, much lower RAM)
      - stream parsed output to parquet (bounded memory per worker)
      - cap workers by default to avoid OOM from many spaCy processes
      - stream spaCy inputs (no big intermediate make_spacy_input DataFrame)
    """
    decade = str(decade)

    if operating_system is None:
        operating_system = platform.system()

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    # Safety: too many spaCy processes will OOM on big decades.
    if worker_cap is not None and workers > worker_cap:
        print(f"[{decade}] Capping workers from {workers} -> {worker_cap} to avoid OOM with spaCy multiprocessing.")
        workers = worker_cap

    # Ensure stable IDs
    decade_df = ensure_speech_ids(decade_subset, decade=decade)
    chunks = make_chunks(decade_df, num_chunks)

    # 1) Write base chunks sequentially (avoids pickling huge DF into workers)
    for chunk_id in range(1, num_chunks + 1):
        df_chunk = chunks.get(chunk_id)
        _write_base_chunk(df_chunk, chunks_dir=chunks_dir, decade=decade, chunk_id=chunk_id)

    # 2) Parallelize parsing (only for missing parsed chunks)
    to_parse: List[int] = []
    for chunk_id in range(1, num_chunks + 1):
        if not _parsed_chunk_path(chunks_dir, decade, chunk_id).exists():
            to_parse.append(chunk_id)

    if not to_parse:
        print(f"[{decade}] All parsed chunks already exist; skipping parse.")
        return

    print(f"[{decade}] Parsing {len(to_parse)}/{num_chunks} chunks with workers={workers} ...")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(spacy_model,),
    ) as ex:
        futures = [
            ex.submit(
                _parse_chunk_worker,
                decade=decade,
                chunk_id=chunk_id,
                chunks_dir=str(chunks_dir),
                max_chars=max_chars,
                split_threshold_chars=split_threshold_chars,
                spacy_model=spacy_model,
                spacy_batch_size=spacy_batch_size,
                token_rows_buffer=token_rows_buffer,
            )
            for chunk_id in to_parse
        ]

        for fut in as_completed(futures):
            _ = fut.result()  # raises if worker errored

    print(f"[{decade}] Done parsing.")