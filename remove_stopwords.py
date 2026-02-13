# remove_stopwords.py

from __future__ import annotations

import pandas as pd

DEFAULT_STOPWORDS_URL = (
    "https://raw.githubusercontent.com/stephbuon/text-mine-congress/main/analysis/congress_stopwords.csv"
)


def load_congress_stopwords(url: str = DEFAULT_STOPWORDS_URL) -> set[str]:
    """
    R equivalent:
      read_csv(url, col_names=FALSE) %>% pull(1) %>% str_trim() %>% discard(~ .x == "")
      stopwords_set <- unique(tolower(congress_stopwords))
    """
    s = pd.read_csv(url, header=None)[0].astype(str)
    s = s.str.strip()
    s = s[s != ""]
    return set(s.str.lower().unique())


def remove_stopwords(df: pd.DataFrame, stopwords_set: set[str]) -> pd.DataFrame:
    """
    Python equivalent of remove_stopwords.R.

    Assumes a column named 'token' exists.
    Steps:
      - lowercase token
      - drop if token in stopwords_set
      - drop if token is ONLY punctuation
      - drop if token contains any digit
      - drop if nchar(token) <= 1
      - drop if trimmed token == ""
    """
    if "token" not in df.columns:
        raise KeyError("DataFrame must contain a 'token' column")

    out = df.copy()
    tok = out["token"].astype(str).str.lower()
    tok_stripped = tok.str.strip()

    # build mask of rows to keep
    keep = pd.Series(True, index=out.index)

    # token in stopwords
    keep &= ~tok.isin(stopwords_set)

    # token is ONLY punctuation (approx of R's ^[[:punct:]]+$)
    # This matches strings made entirely of non-word, non-space chars.
    keep &= ~tok_stripped.str.fullmatch(r"[^\w\s]+", na=False)

    # contains any digit
    keep &= ~tok_stripped.str.contains(r"\d", na=False)

    # length > 1
    keep &= tok_stripped.str.len().fillna(0).astype(int) > 1

    # not empty after trimming
    keep &= tok_stripped.ne("")

    # apply and set lowered token (R mutates token := str_to_lower(token))
    out = out.loc[keep].copy()
    out["token"] = tok_stripped.loc[keep].values

    return out
