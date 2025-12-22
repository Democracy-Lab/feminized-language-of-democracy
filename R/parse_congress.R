#plan(multisession, workers = parallelly::availableCores(omit = 1))

# If your chunks are large, increase the allowed globals size (adjust to your RAM)
#options(future.globals.maxSize = 8 * 1024^3)  # 8GB


#log_message <- function(...) {
#  log_dir <- "logs"
#  log_file <- file.path(log_dir, "logfile.txt")
  
#  if (!dir.exists(log_dir)) {
#    dir.create(log_dir) }
  
#  con <- file(log_file, open = "a")
#  on.exit(close(con), add = TRUE)
  
#  writeLines(paste0(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), paste(..., collapse = " ")), con) }

#split_long_text <- function(text, max_chars = 900000) {
#  n <- nchar(text)
#  if (is.na(n) || n <= max_chars) return(list(text))
  
#  starts <- seq(1, n, by = max_chars)
#  ends   <- pmin(starts + max_chars - 1, n)
  
#  Map(function(s, e) substr(text, s, e), starts, ends) }


#make_spacy_input <- function(original_df, max_chars = 900000) {
#  original_df %>%
#    select(doc_id, content) %>%
#    mutate(piece_list = lapply(content, 
#                               split_long_text, 
#                               max_chars = max_chars)) %>%
#    unnest_longer(piece_list) %>%
#    group_by(doc_id) %>%
#    mutate(piece_id = row_number()) %>%
#    ungroup() %>%
#    transmute(doc_id = paste0(doc_id, "_p", piece_id),
#              text  = piece_list) }


#spacy_parse_unix <- function(df, chunk_id, current_decade, num_chunks) {
  
#  message(paste0("Worker starting for chunk ", chunk_id))
  
#  original_df <- df %>%
#    mutate(doc_id = paste0("text", seq_len(n())))
  
#  input_text <- make_spacy_input(original_df, max_chars = 900000)
  
#  parsed <- spacy_parse(input_text,
#                        pos = TRUE,
#                        dependency = TRUE,
#                        lemma = TRUE,
#                        tag = TRUE, 
#                        entity = FALSE)
  
#  write_parquet(original_df, 
#                sink = file.path("data", "chunks", paste0("us_congress_", current_decade, "_chunk_", chunk_id, ".parquet")))
  
#  tidy_ngrams <- bind_rows(parsed)
  
#  write_parquet(tidy_ngrams, 
#                sink = file.path("data", "chunks", paste0("us_congress_spacy_parsed_", current_decade, "_chunk_", chunk_id, ".parquet")))
  
#  spacy_finalize() }


#spacy_parse_windows <- function(df, chunk_id, current_decade, num_chunks) {
  
#  print(paste0("Worker starting for chunk ", chunk_id))
  
#  original_df <- df %>%
#    mutate(doc_id = paste0("text", seq_len(n())))
  
#  input_text <- make_spacy_input(original_df, max_chars = 900000)
  
#  parsed <- spacy_parse(input_text,
#                        pos = TRUE,
#                        dependency = TRUE,
#                        lemma = TRUE,
#                        tag = TRUE, 
#                        entity = FALSE)
  
#  write_parquet(original_df, 
#                sink = file.path("data", "chunks", paste0("us_congress_", current_decade, "_chunk_", chunk_id, ".parquet")))

#  tidy_ngrams <- bind_rows(parsed)
  
#  write_parquet(tidy_ngrams, 
#                sink = file.path("data", "chunks", paste0("us_congress_spacy_parsed_", current_decade, "_chunk_", chunk_id, ".parquet")))
  
#  spacy_finalize() }

#process <- function(data, d, num_chunks, os) {
#  tic(paste0("Total time parallel processing ", d))
  
#  print("Processing Data")
#  print(paste0("Number of Chunks: ", num_chunks))

#  chunks <- data %>%
#    mutate(chunk_id = ntile(row_number(), num_chunks)) %>%
#    group_split(chunk_id)
  
#  if(os=="Windows") {
#    future_map2(chunks, seq_along(chunks),
#                ~ spacy_parse_windows(.x, .y, d, num_chunks), 
#                .options = furrr_options(seed = TRUE)) } # Seed for reproducability. Is it needed? 
  
#  if(os=="Linux") {
#    future_map2(chunks, seq_along(chunks),
#                ~ spacy_parse_unix(.x, .y, d, num_chunks), 
#                .options = furrr_options(seed = TRUE))  } # Seed for reproducability. Is it needed? 
  
  
#  timing <- toc(log = TRUE, quiet = TRUE)
#  elapsed_sec <- timing$toc - timing$tic
#  elapsed_min <- elapsed_sec / 60
#  print(paste("Elapsed time:", round(elapsed_min, 2), "minutes")) 
#  gc() }


# R/parse_congress.R
# Stable disk-first chunking + furrr multisession-safe parsing

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(arrow)
  library(spacyr)
  library(furrr)
  library(future)
  library(parallelly)
  library(tictoc)
  library(tibble)
})

# ---------------------------
# Logging (optional)
# ---------------------------
log_message <- function(...) {
  log_dir <- "logs"
  log_file <- file.path(log_dir, "logfile.txt")
  
  if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE)
  
  con <- file(log_file, open = "a")
  on.exit(close(con), add = TRUE)
  
  writeLines(
    paste0(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), paste(..., collapse = " ")),
    con
  )
}

# ---------------------------
# Step 1: Write chunk parquet files sequentially (no futures)
# ---------------------------
write_chunk_files <- function(data,
                              decade,
                              num_chunks,
                              chunks_dir = file.path("data", "chunks")) {
  if (!dir.exists(chunks_dir)) dir.create(chunks_dir, recursive = TRUE)
  
  message(paste0("[", decade, "] Writing chunk files 1..", num_chunks))
  
  # Deterministic chunking (same as your ntile approach)
  chunks <- data %>%
    dplyr::mutate(chunk_id = dplyr::ntile(dplyr::row_number(), num_chunks)) %>%
    dplyr::group_split(chunk_id)
  
  # Write each chunk as us_congress_<decade>_chunk_<i>.parquet
  for (i in seq_along(chunks)) {
    out_path <- file.path(chunks_dir, paste0("us_congress_", decade, "_chunk_", i, ".parquet"))
    arrow::write_parquet(chunks[[i]], sink = out_path)
  }
  
  invisible(TRUE)
}

# ---------------------------
# Step 2: Parse ONE chunk by reading from disk (runs in worker)
# Most stable pattern on Windows multisession:
# - define helpers inside worker to avoid exporting globals
# - spacy_initialize inside worker
# ---------------------------
parse_chunk_from_disk <- function(chunk_id,
                                  decade,
                                  max_chars = 900000,
                                  chunks_dir = file.path("data", "chunks"),
                                  spacy_model = "en_core_web_sm") {
  
  suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(arrow)
    library(spacyr)
  })
  
  # Helper functions defined INSIDE the worker to avoid huge exported globals
  split_long_text <- function(text, max_chars = 900000) {
    n <- nchar(text)
    if (is.na(n) || n <= max_chars) return(list(text))
    
    starts <- seq(1, n, by = max_chars)
    ends   <- pmin(starts + max_chars - 1, n)
    
    Map(function(s, e) substr(text, s, e), starts, ends)
  }
  
  make_spacy_input <- function(original_df, max_chars = 900000) {
    original_df %>%
      dplyr::select(doc_id, content) %>%
      dplyr::mutate(piece_list = lapply(content, split_long_text, max_chars = max_chars)) %>%
      tidyr::unnest_longer(piece_list) %>%
      dplyr::group_by(doc_id) %>%
      dplyr::mutate(piece_id = dplyr::row_number()) %>%
      dplyr::ungroup() %>%
      dplyr::transmute(
        doc_id = paste0(doc_id, "_p", piece_id),
        text   = piece_list
      )
  }
  
  in_path  <- file.path(chunks_dir, paste0("us_congress_", decade, "_chunk_", chunk_id, ".parquet"))
  out_path <- file.path(chunks_dir, paste0("us_congress_spacy_parsed_", decade, "_chunk_", chunk_id, ".parquet"))
  
  # If the input chunk file somehow doesn't exist, stop clearly
  if (!file.exists(in_path)) {
    stop(paste0("Missing input chunk file: ", in_path))
  }
  
  message(paste0("[", decade, "] Worker parsing chunk ", chunk_id))
  
  df <- arrow::read_parquet(in_path)
  
  # Initialize spaCy inside worker (critical for multisession)
  spacyr::spacy_initialize(model = spacy_model, refresh_settings = TRUE)
  
  original_df <- df %>%
    dplyr::mutate(doc_id = paste0("text", seq_len(n())))
  
  input_text <- make_spacy_input(original_df, max_chars = max_chars)
  
  parsed <- spacyr::spacy_parse(
    input_text,
    pos = TRUE,
    dependency = TRUE,
    lemma = TRUE,
    tag = TRUE,
    entity = FALSE
  )
  
  tidy_ngrams <- dplyr::bind_rows(parsed)
  arrow::write_parquet(tidy_ngrams, sink = out_path)
  
  spacyr::spacy_finalize()
  invisible(TRUE)
}

# ---------------------------
# Main entry point: disk-first chunking + parallel parse over IDs
# ---------------------------
process <- function(data,
                    d,
                    num_chunks,
                    os,
                    max_chars = 900000,
                    chunks_dir = file.path("data", "chunks"),
                    spacy_model = "en_core_web_sm") {
  
  tictoc::tic(paste0("Total time processing ", d))
  message(paste0("[", d, "] Processing. num_chunks=", num_chunks, " os=", os))
  
  # 1) Always write chunk files sequentially first
  write_chunk_files(data, decade = d, num_chunks = num_chunks, chunks_dir = chunks_dir)
  
  # 2) Build a parameter table and p-walk it
  # This avoids anonymous functions capturing big environments.
  params <- tibble::tibble(
    chunk_id   = seq_len(num_chunks),
    decade     = d,
    max_chars  = max_chars,
    chunks_dir = chunks_dir,
    spacy_model = spacy_model
  )
  
  # 3) Parallel parse by chunk_id ONLY (tiny globals)
  # globals=FALSE prevents furrr exporting giant closures (your 4.13 GiB ...furrr_fn)
  furrr::future_pwalk(
    params,
    parse_chunk_from_disk,
    .options = furrr::furrr_options(
      seed = TRUE,
      globals = FALSE,
      packages = c("dplyr", "tidyr", "arrow", "spacyr")
    )
  )
  
  timing <- tictoc::toc(log = TRUE, quiet = TRUE)
  elapsed_sec <- timing$toc - timing$tic
  message(paste0("[", d, "] Done in ", round(elapsed_sec / 60, 2), " minutes"))
  
  gc()
  invisible(TRUE)
}

