

#PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) {
#  normalizePath(here::here(), winslash = "/", mustWork = TRUE)
#} else {
#  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
#}

#DATA_DIR   <- file.path(PROJECT_ROOT, "data")
#chunks_dir <- file.path(DATA_DIR, "chunks")

#stopifnot(dir.exists(DATA_DIR))
#if (!dir.exists(chunks_dir)) dir.create(chunks_dir, recursive = TRUE)



existing_chunk_ids <- function(decade) {
  files <- list.files(chunks_dir,
                      pattern = paste0("^us_congress_", decade, "_chunk_\\d+\\.parquet$"),
                      full.names = FALSE)
  ids <- as.integer(str_match(files, "_chunk_(\\d+)\\.parquet$")[,2])
  sort(unique(ids[!is.na(ids)])) }

existing_parsed_ids <- function(decade) {
  files <- list.files(chunks_dir,
                      pattern = paste0("^us_congress_spacy_parsed_", decade, "_chunk_\\d+\\.parquet$"),
                      full.names = FALSE)
  ids <- as.integer(str_match(files, "_chunk_(\\d+)\\.parquet$")[,2])
  sort(unique(ids[!is.na(ids)])) }



make_chunks <- function(decade_df, num_chunks) {
  decade_df %>%
    mutate(chunk_id = ntile(row_number(), num_chunks)) %>%
    group_split(chunk_id) }



run_chunk_index <- function(chunks, decade, chunk_id, os) {
  message("Backfilling chunk_id ", chunk_id)
  
  df_chunk <- chunks[[chunk_id]]
  
  spacy_initialize(model = "en_core_web_sm", refresh_settings = TRUE)
  
  if (os == "Windows") {
    spacy_parse_windows(df_chunk, chunk_id, decade, num_chunks = length(chunks))
  } else {
    spacy_parse_unix(df_chunk, chunk_id, decade, num_chunks = length(chunks)) } }

# ----------------------------
# Main backfill function
# ----------------------------
backfill_missing_chunks <- function(decade_file_paths,
                                    decades = NULL,
                                    num_chunks = 800,
                                    os = Sys.info()[["sysname"]],
                                    print_existing = TRUE,
                                    print_missing_parsed_too = TRUE) {
  
  if (is.null(decades)) {
    decades <- str_extract(basename(decade_file_paths), "\\d{4}") }
  
  for (decade in decades) {
    
    print("Decade:", decade)

    f <- decade_file_paths[str_detect(basename(decade_file_paths),
                                      paste0("^us_congress_", decade, "\\.parquet$"))]
    
    if (length(f) != 1) {
      stop("Could not uniquely identify decade parquet for ", decade) }
    
    have_chunks  <- existing_chunk_ids(decade)
    target_ids   <- seq_len(num_chunks)
    missing_base <- setdiff(target_ids, have_chunks)
    
    if (print_existing) {
      cat("Existing base chunk_ids (", length(have_chunks), "):\n", sep = "")
      print(have_chunks) }
    
    cat("\nBackfilling BASE chunk_ids (", length(missing_base), "):\n", sep = "")
    if (length(missing_base) > 0) print(missing_base) else cat("None 🎉\n")
    
    if (print_missing_parsed_too) {
      have_parsed <- existing_parsed_ids(decade)
      missing_parsed <- setdiff(target_ids, have_parsed)
      
      cat("\nMissing PARSED chunk_ids (", length(missing_parsed), "):\n", sep = "")
      if (length(missing_parsed) > 0) print(missing_parsed) else cat("None") }
    
    if (length(missing_base) == 0 &&
        (!print_missing_parsed_too || length(missing_parsed) == 0)) {
      next }
    
    cat("\nLoading decade data from:\n", f, "\n", sep = "")
    decade_df <- read_parquet(f)
    
    cat("Rebuilding chunk list (num_chunks =", num_chunks, ")\n")
    chunks <- make_chunks(decade_df, num_chunks)
    
    # ---- PRINT again right before execution ----
    if (length(missing_base) > 0) {
      cat("\nExecuting backfill for BASE chunk_ids:\n")
      print(missing_base)
      
      for (cid in missing_base) {
        run_chunk_index(chunks, decade, cid, os) } }
    
    if (print_missing_parsed_too) {
      have_parsed_after <- existing_parsed_ids(decade)
      missing_parsed_after <- setdiff(target_ids, have_parsed_after)
      
      if (length(missing_parsed_after) > 0) {
        cat("\nExecuting backfill for PARSED-only chunk_ids:\n")
        print(missing_parsed_after)
        
        for (cid in missing_parsed_after) {
          run_chunk_index(chunks, decade, cid, os)
        }
      } else {
        print("All parsed chunks exist after base backfill")
      }
    }
    
    print("Done decade", decade)
  }
  
  invisible(TRUE) }




 backfill_missing_chunks(
   decade_file_paths = file_list,
   decades = "1950",
   num_chunks = 800,
   os = operating_system
 )

#backfill_missing_chunks(
#  decade_file_paths = file_list,
#  decades = c("1900", "1910", "1920", "1930", "1940", "1950"),
#  num_chunks = 800,
#  os = operating_system
#)
