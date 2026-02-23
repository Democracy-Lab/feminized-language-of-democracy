#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(dplyr)
  library(tidyr)
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
LEMMA_ROOT_HARDCODED <- "lemma_outputs/lemma_cache_2026-02-16_113305_top50"

# ---- CLI ----
parser <- ArgumentParser(description = "Visualize shared vocab overlap/jaccard from cached lemma frequencies.")
parser$add_argument("--lemma_root", default = LEMMA_ROOT_HARDCODED,
                    help = "Root folder containing <category>/<decade>/lemma_freq.parquet (default is hardcoded)")
parser$add_argument("--output_dir", default = "outputs", help = "Directory for all outputs")
parser$add_argument("--top_n", type="integer", default = 50, help = "Top-N terms per gender/decade")
args <- parser$parse_args()

lemma_root <- normalizePath(args$lemma_root, mustWork = TRUE)
out_dir <- args$output_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOP_N <- args$top_n

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

  # NEW: collapse identical lemmas across POS (or any duplicate rows) within what is already saved
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
# Core loop (same outputs as your original)
# -----------------------------
percat_counts <- list()
percat_shared_terms <- list()
percat_jaccard <- list()

for (cat in unique(meta_df$category)) {
  cat_files <- meta_df |> dplyr::filter(category == cat)

  rows_counts <- list()
  rows_shared_terms <- list()
  rows_jaccard <- list()

  for (i in seq_len(nrow(cat_files))) {
    dec  <- cat_files$decade[i]
    path <- cat_files$file[i]

    t0 <- Sys.time()
    message(sprintf("[%s] START cache: category=%s decade=%s path=%s",
                    format(t0, "%Y-%m-%d %H:%M:%S"), cat, dec, basename(path)))

    tops <- get_top_terms_by_gender_from_cache(path, TOP_N)
    top_m <- tops$M
    top_f <- tops$F

    message(sprintf("[%s] INFO  got_top_terms (M=%d, F=%d)",
                    format(Sys.time(), "%Y-%m-%d %H:%M:%S"), length(top_m), length(top_f)))

    shared <- intersect(top_m, top_f)
    men_only <- setdiff(top_m, shared)
    women_only <- setdiff(top_f, shared)

    rows_counts[[length(rows_counts)+1]] <- tibble(
      category = cat, decade = dec,
      membership = c("Men only","Shared","Women only"),
      count = c(length(men_only), length(shared), length(women_only))
    )

    if (length(shared) > 0) {
      rows_shared_terms[[length(rows_shared_terms)+1]] <- tibble(
        category = cat, decade = dec, term = shared
      )
    }

    u <- union(top_m, top_f)
    j <- if (length(u) > 0) length(shared) / length(u) else 0

    rows_jaccard[[length(rows_jaccard)+1]] <- tibble(
      category = cat, decade = dec, jaccard = j,
      shared_frac = ifelse(TOP_N > 0, length(shared)/TOP_N, NA_real_)
    )

    t1 <- Sys.time()
    message(sprintf("[%s] DONE  cache: category=%s decade=%s elapsed=%s shared=%d jaccard=%.4f",
                    format(t1, "%Y-%m-%d %H:%M:%S"),
                    cat, dec, format(t1 - t0), length(shared), j))
  }

  percat_counts[[cat]] <- dplyr::bind_rows(rows_counts)
  percat_shared_terms[[cat]] <- dplyr::bind_rows(rows_shared_terms)
  percat_jaccard[[cat]] <- dplyr::bind_rows(rows_jaccard)
}

counts_df <- dplyr::bind_rows(percat_counts)
shared_terms_df <- dplyr::bind_rows(percat_shared_terms)
jaccard_df <- dplyr::bind_rows(percat_jaccard)

# ---- Save tables (identical filenames) ----
readr::write_csv(counts_df, file.path(out_dir, sprintf("overlap_counts_top%d.csv", TOP_N)))
readr::write_csv(shared_terms_df, file.path(out_dir, sprintf("shared_terms_top%d.csv", TOP_N)))
readr::write_csv(jaccard_df, file.path(out_dir, sprintf("jaccard_top%d.csv", TOP_N)))

# ---- Plots (same as your original) ----
plot_counts <- function(df_cat, cat) {
  p <- ggplot(df_cat, aes(decade, count, fill = membership)) +
    geom_col() +
    scale_fill_manual(values = c("Men only" = "#56B4E9", "Shared" = "#999999", "Women only" = "#E69F00")) +
    labs(title = paste0("Overlap of Top-", TOP_N, " Words by Gender & Decade — ", gsub("_"," ",cat)),
         x = "Decade", y = "Number of Words", fill = NULL) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(angle=30, hjust=1), legend.position = "top")
  ggsave(file.path(out_dir, sprintf("overlap_top%d_%s.png", TOP_N, cat)), p,
         width = max(8, length(unique(df_cat$decade))*1.2), height=6, dpi=300, bg="white")
}

plot_jaccard <- function(df_cat, cat) {
  p <- ggplot(df_cat, aes(decade, jaccard, group=1)) +
    geom_line() + geom_point() +
    scale_y_continuous(labels=scales::percent_format(accuracy=1)) +
    labs(title = paste0("Men vs Women Top-", TOP_N, " Similarity (Jaccard) — ", gsub("_"," ",cat)),
         x = "Decade", y = "Similarity (% of union)") +
    theme_minimal(base_size = 13) + theme(axis.text.x = element_text(angle=30, hjust=1))
  ggsave(file.path(out_dir, sprintf("jaccard_top%d_%s.png", TOP_N, cat)), p,
         width = max(8, length(unique(df_cat$decade))*1.2), height=5, dpi=300, bg="white")
}

plot_shared_leaderboard <- function(shared_df_cat, cat, k = 25) {
  if (nrow(shared_df_cat) == 0) return(invisible(NULL))
  freq <- shared_df_cat |>
    count(term, sort=TRUE) |>
    dplyr::slice_head(n = k) |>
    arrange(n, term)
  p <- ggplot(freq, aes(n, reorder(term, n))) +
    geom_col() +
    labs(title = paste0("Most-Shared Terms across Decades — ", gsub("_"," ",cat)),
         x = "# of decades where term is shared", y = NULL) +
    theme_minimal(base_size = 13)
  ggsave(file.path(out_dir, sprintf("shared_terms_leaderboard_top%d_%s.png", TOP_N, cat)), p,
         width=8, height=max(6, 0.3*nrow(freq)+1), dpi=300, bg="white")
}


for (cat in unique(meta_df$category)) {
  dfc <- counts_df |> dplyr::filter(category==cat)
  if (nrow(dfc)>0) plot_counts(dfc, cat)

  dfj <- jaccard_df |> dplyr::filter(category==cat)
  if (nrow(dfj)>0) plot_jaccard(dfj, cat)

  dfs <- shared_terms_df |> dplyr::filter(category==cat)
  if (nrow(dfs)>0) plot_shared_leaderboard(dfs, cat)
}

message("Done. Tables in ", normalizePath(out_dir), "; figures saved alongside.")
