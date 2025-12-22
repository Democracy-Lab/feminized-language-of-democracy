library(here)
library(tidyverse)
library(arrow)
library(data.table)
library(furrr)
library(future)
library(tictoc)
library(spacyr)
library(reticulate)
library(lubridate)

debug <- FALSE
should_delete_chunks <- FALSE
decades <- c(1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020)

data_dir <- here("data")
if (!dir.exists(data_dir)) {
  dir.create(data_dir) 
  message("Created data directory") } else {
    message("Data directory already exists") }

chunks_dir <- file.path("data", "chunks")
if (!dir.exists(chunks_dir)) {
  dir.create(chunks_dir) 
  message("Created chunks directory") } else {
    message("Chunks directory already exists") }

gender_dir <- file.path("data", "gender_analysis")
if (!dir.exists(gender_dir)) {
  dir.create(gender_dir) 
  message("Created gender analysis directory") } else {
    message("Gender analysis directory already exists") }

categories_dir <- file.path("data", "gender_analysis", "categories")
if (!dir.exists(categories_dir)) {
  dir.create(categories_dir) 
  message("Created categories directory in gender analysis directory") } else {
    message("Categories directory inside gender analysis directory already exists") }

# Start Data processing Pipeline

source("R/create_decade_subset.R")

us_congress_data <- fread(file.path(data_dir, "combined_congressional_records_12-07-25.csv")) #"congress_data_daily_by_speaker_with_metadata.csv"))
us_congress_data <- create_decades_col(us_congress_data)

if(debug==TRUE) {
  us_congress_data <- us_congress_data %>%
    group_by(decade) %>%
    slice_sample(n = 100) %>%
    ungroup() }

split_by_decade(us_congress_data, data_dir)

source("R/parse_congress.R")

operating_system <- Sys.info()["sysname"]

if(operating_system == "Windows") {
  #use_python("C:/Users/steph/AppData/Local/R/cache/R/reticulate/uv/cache/archive-v0/CVOhsXrYEbrjpDEIQ24Ag/Scripts/python.exe", required = TRUE)
  spacy_initialize(model = "en_core_web_sm", refresh_settings = TRUE)
  plan(multisession, workers = 20) }

if(operating_system == "Linux") {
  spacy_initialize(model = "en_core_web_sm", refresh_settings = TRUE)
  plan(multicore, workers = 22) } 

file_list <- list.files(path = "data", pattern = "^us_congress_\\d{4}\\.parquet$", full.names = TRUE)

for (f in file_list) {
  print(paste0("Loading: ", f))
  print(paste0("Number of Workers (Cores): ", nbrOfWorkers()))
  decade_subset <- read_parquet(f) 
  current_decade <- str_extract(f, "\\d{4}")
  process(decade_subset, current_decade, num_chunks = 800, operating_system) 
  
  # Add backfill here
  # Right now, not using it -- but add here if want to use 
  
  } 

source("R/bind_data.R")

bind_chunks(chunk_type = "spacy_parse")
bind_chunks(chunk_type = "default")

if(should_delete_chunks == TRUE) {
  delete_chunks() }

source("R/subset_by_gender.R")

for(d in decades) {
  subset_by_gender(d, data_dir) }

source("R/remove_stopwords.R")

for(d in decades) {
  men_file_name <- paste0("us_congress_men_", d, ".parquet")
  women_file_name <- paste0("us_congress_women_", d, ".parquet")
  
  parsed_decade_subset_men <- read_parquet(file.path(gender_dir, men_file_name))
  parsed_decade_subset_women <- read_parquet(file.path(gender_dir, women_file_name)) 
  
  out1 <- remove_stopwords(parsed_decade_subset_men, stopwords_set)
  out2 <- remove_stopwords(parsed_decade_subset_women, stopwords_set)
 
  all_data <- bind_rows(out1, out2)
  
  write_parquet(all_data, 
                sink = file.path(gender_dir, paste0("us_congress_men_and_women_clean_", d, ".parquet"))) 
  
  rm(parsed_decade_subset_men, parsed_decade_subset_women, out1, out2, all_data)
  gc() }


source("R/subset_by_category.R")

for(d in decades) { 
  f <- read_parquet(file.path(gender_dir, paste0("us_congress_men_and_women_clean_", d, ".parquet")))
  subset_by_category(f, d, categories_dir)
  gc() }

