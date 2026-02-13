from __future__ import annotations

from pathlib import Path
import pandas as pd


def create_decades_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - year (nullable Int64)
      - decade (nullable Int64)

    IMPORTANT: decade is kept as numeric NA (pd.NA) when date parsing fails,
    so we do NOT accidentally create a file like us_congress_<NA>.parquet.
    """
    out = df.copy()

    # drop granule_id if present
    if "granule_id" in out.columns:
        out = out.drop(columns=["granule_id"])

    dates = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = dates.dt.year.astype("Int64")          # keep nullable integer
    out["decade"] = ((out["year"] // 10) * 10).astype("Int64")  # keep nullable integer

    return out


def split_by_decade(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Writes: <output_dir>/us_congress_<decade>.parquet

    Deterministic ordering:
      - If df lacks 'global_row_id', create it as 1..N in the current df order
      - Sort within each decade by global_row_id (stable)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add global_row_id if missing (deterministic within this run)
    if "global_row_id" not in df.columns:
        df = df.copy()
        df["global_row_id"] = range(1, len(df) + 1)

    # Drop rows with missing decade so we don't generate "<NA>" decade files
    df_nonnull = df[df["decade"].notna()].copy()

    decades = sorted(df_nonnull["decade"].unique().tolist())

    for d in decades:
        d_int = int(d)  # safe because we dropped NA
        print(f"Subsetting {d_int}")

        decade_subset = df_nonnull[df_nonnull["decade"] == d].copy()

        # deterministic row order within decade
        decade_subset = decade_subset.sort_values("global_row_id", kind="stable")

        decade_name = f"us_congress_{d_int}"
        out_path = output_dir / f"{decade_name}.parquet"
        decade_subset.to_parquet(out_path, index=False)
