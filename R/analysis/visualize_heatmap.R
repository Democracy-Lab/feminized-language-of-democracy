#!/usr/bin/env Rscript

# Make sure we can see both of your user libraries
.libPaths(c(
  path.expand("~/R/library"),
  path.expand("~/R/x86_64-pc-linux-gnu-library/4.4"),
  .libPaths()
))

suppressPackageStartupMessages({
  library(argparse)
  library(dplyr)
  library(stringr)
  library(purrr)
  library(readr)
  library(ggplot2)
  library(data.table)
  library(scales)
  library(arrow)     # read_parquet
})

# -----------------------------
# HARD-CODE THIS after you run Python once:
# -----------------------------
LEMMA_ROOT_HARDCODED <- "lemma_outputs/lemma_cache_2026-02-16_122318_top50"
# ^^^ change to the folder printed by lemmatize_cache.py

# ---- CLI ----
parser <- ArgumentParser(description = "Heatmap of Men vs Women similarity using Rank-Biased Overlap (RBO) from cached lemma frequencies.")
parser$add_argument("--lemma_root", default = LEMMA_ROOT_HARDCODED,
                    help = "Root folder containing <category>/<decade>/lemma_freq.parquet (default is hardcoded)")
parser$add_argument("--output_dir", default = "heatmaps", help = "Directory for all outputs")
parser$add_argument("--top_n", type="integer", default = 50, help = "Top-N terms per gender/decade used to compute RBO")

# IMPORTANT: do NOT use NA_real_ as a default; argparse shells out to python and will crash on NA.
parser$add_argument("--rbo_p", type="double", default = NULL,
                    help = "RBO weight parameter p in (0,1). Default: 1 - 1/top_n (e.g., top_n=50 -> p=0.98).")

args <- parser$parse_args()

lemma_root <- normalizePath(args$lemma_root, mustWork = TRUE)
out_dir <- args$output_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOP_N <- args$top_n

# default p = 1 - 1/top_n, unless user explicitly sets --rbo_p
RBO_P <- args$rbo_p
if (is.null(RBO_P) || is.na(RBO_P)) {
  if (TOP_N <= 1) {
    stop("top_n must be > 1 when --rbo_p is not provided (so default 1 - 1/top_n is valid).")
  }
  RBO_P <- 1 - (1 / TOP_N)
}

# validate p
if (!is.finite(RBO_P) || RBO_P <= 0 || RBO_P >= 1) {
  stop("--rbo_p must be a finite number strictly between 0 and 1. Got: ", RBO_P)
}

# -----------------------------
# Discover cached parquet files
# -----------------------------
parquets <- list.files(lemma_root, pattern = "lemma_freq\\.parquet$", recursive = TRUE, full.names = TRUE)
if (length(parquets) == 0) stop("No lemma_freq.parquet files found under lemma_root: ", lemma_root)

# Each parquet path looks like: <lemma_root>/<category>/<decade>/lemma_freq.parquet
parse_cache_meta <- function(path) {
  rel <- gsub(paste0("^", gsub("\\\\","/", lemma_root), "/?"), "", gsub("\\\\","/", path))
  parts <- strsplit(rel, "/")[[1]]
  if (length(parts) < 3) return(NULL)
  list(category = parts[1], decade = parts[2], file = path)
}

meta <- purrr::map(parquets, parse_cache_meta) |> purrr::compact()
meta_df <- purrr::map_df(meta, ~tibble(category=.x$category, decade=.x$decade, file=.x$file)) |>
  arrange(category, decade)

if (nrow(meta_df) == 0) stop("Could not parse cache folder structure under: ", lemma_root)

# -----------------------------
# Read one cache and get top terms
# -----------------------------
get_top_terms_by_gender_from_cache <- function(parquet_path, TOP_N) {
  freq <- arrow::read_parquet(parquet_path, as_data_frame = TRUE)
  if (nrow(freq) == 0) return(list(M=character(0), F=character(0)))
  freq <- as.data.table(freq)

  # hard guarantee expected cols
  if (!all(c("gender","lemma","N") %in% names(freq))) return(list(M=character(0), F=character(0)))
  freq[, gender := as.character(gender)]
  freq[, lemma := as.character(lemma)]
  freq[, N := as.integer(N)]

  # collapse identical lemmas across POS (or any duplicate rows) within what is already saved
  freq <- freq[, .(N = sum(N, na.rm = TRUE)), by = .(gender, lemma)]

  top_for_gender <- function(g) {
    x <- freq[gender == g]
    if (nrow(x) == 0) return(character(0))
    setorder(x, -N, lemma)
    head(x$lemma, TOP_N)
  }
  list(M = top_for_gender("M"), F = top_for_gender("F"))
}

# -----------------------------
# Rank-Biased Overlap (RBO)
# -----------------------------
rbo_ext <- function(S, T, p = RBO_P) {
  S <- as.character(S); T <- as.character(T)
  if (length(S) == 0 && length(T) == 0) return(NA_real_)
  if (length(S) == 0 || length(T) == 0) return(0)

  # ensure no duplicates while preserving rank order (defensive)
  S <- S[!duplicated(S)]
  T <- T[!duplicated(T)]

  k <- max(length(S), length(T))
  sum_term <- 0.0

  seenS <- new.env(parent = emptyenv())
  seenT <- new.env(parent = emptyenv())

  overlap <- 0L

  for (d in seq_len(k)) {
    if (d <= length(S)) {
      s <- S[d]
      assign(s, TRUE, envir = seenS)
      if (exists(s, envir = seenT, inherits = FALSE)) overlap <- overlap + 1L
    }
    if (d <= length(T)) {
      t <- T[d]
      assign(t, TRUE, envir = seenT)
      if (exists(t, envir = seenS, inherits = FALSE)) overlap <- overlap + 1L
    }

    sum_term <- sum_term + (overlap / d) * (p^(d - 1))
  }

  rbo <- (1 - p) * sum_term + (overlap / k) * (p^k)
  max(0, min(1, rbo))
}

# -----------------------------
# Compute RBO per category/decade
# -----------------------------
rows_rbo <- list()

for (i in seq_len(nrow(meta_df))) {
  cat <- meta_df$category[i]
  dec <- meta_df$decade[i]
  path <- meta_df$file[i]

  t0 <- Sys.time()
  message(sprintf("[%s] START cache: category=%s decade=%s path=%s",
                  format(t0, "%Y-%m-%d %H:%M:%S"), cat, dec, basename(path)))

  tops <- get_top_terms_by_gender_from_cache(path, TOP_N)
  top_m <- tops$M
  top_f <- tops$F

  rbo <- rbo_ext(top_m, top_f, p = RBO_P)

  rows_rbo[[length(rows_rbo) + 1]] <- tibble(
    category = cat,
    decade = dec,
    rbo = rbo,
    top_n = TOP_N,
    rbo_p = RBO_P
  )

  t1 <- Sys.time()
  message(sprintf("[%s] DONE  cache: category=%s decade=%s elapsed=%s rbo=%.4f",
                  format(t1, "%Y-%m-%d %H:%M:%S"),
                  cat, dec, format(t1 - t0), ifelse(is.na(rbo), NaN, rbo)))
}

rbo_df <- dplyr::bind_rows(rows_rbo)

# ---- Save table ----
readr::write_csv(rbo_df, file.path(out_dir, sprintf("rbo_top%d_p%.5f.csv", TOP_N, RBO_P)))

# -----------------------------
# Heatmap
# -----------------------------
heatmap_rbo <- function(df_in) {
  df <- df_in |>
    select(category, decade, rbo)

  df$decade <- factor(df$decade, levels = sort(unique(df$decade)))

  cat_order <- df |>
    group_by(category) |>
    summarize(mean_rbo = mean(rbo, na.rm = TRUE), .groups="drop") |>
    arrange(mean_rbo) |>
    pull(category)

  df$category <- factor(df$category, levels = cat_order)

  p <- ggplot(df, aes(decade, category, fill = rbo)) +
    geom_tile() +
    scale_fill_gradient(low="#f0f0f0", high="#08519c", na.value = "#f7f7f7",
                        labels=scales::percent_format(accuracy=1)) +
    labs(title = paste0("Rank-Biased Overlap (RBO) of Top-", TOP_N, " by Decade × Category (p=", sprintf("%.5f", RBO_P), ")"),
         fill = "RBO", x=NULL, y=NULL) +
    theme_minimal(base_size=12) +
    theme(axis.text.x = element_text(angle=30, hjust=1))

  ggsave(file.path(out_dir, sprintf("heatmap_rbo_top%d_p%.5f.png", TOP_N, RBO_P)), p,
         width = max(10, length(unique(df$decade))*1.0),
         height = max(6, 0.5*length(unique(df$category))), dpi=300, bg="white")
}

if (nrow(rbo_df) > 0) heatmap_rbo(rbo_df)

message("Done. RBO table in ", normalizePath(out_dir), "; heatmap saved alongside.")