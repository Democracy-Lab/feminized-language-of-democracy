#!/usr/bin/env Rscript

# ══════════════════════════════════════════════════════════════════════════════
# Visualize Distinctive Words from Pre-computed CSVs
# FIXES:
# - Correct word labels (no doubling / missing labels) by avoiding coord_flip()
#   with scale_x_reordered(); instead use y=reorder_within(...) + scale_y_reordered()
# - No "gaps" from shared ordering across Men/Women: reorder within each panel key
# - Row facet labels (category/decade) rotated sideways for space
# - For gender×(category/decade) views: build separate Men/Women plots side-by-side
#   so each gender has its own word list (facet_grid only draws one outer axis)
# - Added by_decade visualizations
# - Single row labels (left side only) for gender plots
# - Color coding: Women = red, Men = blue (no grey bars)
# - Neutral colors for by_category and by_decade visualizations
# - Scaled category titles based on length
# - Filter out categories/decades with fewer than 20 words for women
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(tidytext)
  library(stringr)
  library(patchwork)  # <-- needed for side-by-side Men/Women panels
})

# ── CLI Arguments ─────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, flag, default = NULL) {
  for (i in seq_along(args)) {
    a <- args[[i]]
    pattern <- paste0("^--", flag, "=")
    if (grepl(pattern, a)) {
      return(sub(pattern, "", a))
    }
    if (a == paste0("--", flag) && i < length(args)) {
      return(args[[i + 1]])
    }
  }
  return(default)
}

INPUT_DIR <- parse_arg(args, "input")
CSV_ONLY <- !is.null(parse_arg(args, "csv-only"))
MIN_WORDS_ARG <- parse_arg(args, "min-words")

if (is.null(INPUT_DIR)) {
  stop("Must specify --input directory containing computed CSV files")
}

INPUT_DIR <- normalizePath(INPUT_DIR, mustWork = TRUE)

# MIN_WORDS will be set later based on top_n if not provided
MIN_WORDS <- if (!is.null(MIN_WORDS_ARG)) as.integer(MIN_WORDS_ARG) else NULL

message("═══════════════════════════════════════════════════════════════")
message("Input directory: ", INPUT_DIR)
if (CSV_ONLY) {
  message("Mode: CSV ONLY (skipping PNG generation)")
}
if (!is.null(MIN_WORDS)) {
  message("Minimum words for women filter: ", MIN_WORDS)
}
message("═══════════════════════════════════════════════════════════════\n")

# ── Helper Functions ──────────────────────────────────────────────────────────
format_category <- function(cat) str_replace_all(cat, "_", " ")
format_decade   <- function(dec) paste0(dec, "s")

save_plot_auto_height <- function(plot, path, n_terms, n_facets = 1, width = 8) {
  base_height <- 2
  per_term <- 0.18
  height <- base_height + (per_term * n_terms * n_facets)
  height <- min(height, 200)
  ggsave(path, plot, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
  message("  → Saved: ", basename(path))
}

# Safer top_n inference: how many rows per group?
infer_top_n <- function(df, group_cols) {
  df %>%
    count(across(all_of(group_cols))) %>%
    summarise(top_n = max(n, na.rm = TRUE)) %>%
    pull(top_n)
}

# Helper function to wrap or truncate category labels
format_category_label <- function(text, max_chars = 25) {
  if (nchar(text) <= max_chars) {
    return(text)
  } else {
    # Try to wrap at word boundary
    words <- strsplit(text, " ")[[1]]
    lines <- character()
    current_line <- ""
    
    for (word in words) {
      if (nchar(current_line) == 0) {
        current_line <- word
      } else if (nchar(paste(current_line, word)) <= max_chars) {
        current_line <- paste(current_line, word)
      } else {
        lines <- c(lines, current_line)
        current_line <- word
      }
    }
    if (nchar(current_line) > 0) {
      lines <- c(lines, current_line)
    }
    
    return(paste(lines, collapse = "\n"))
  }
}

# ── POS label helpers (infer from CSV filename) ───────────────────────────────
# Expected basenames from your Python script:
# - distinctive_words_by_category_top10.csv
# - distinctive_noun_by_category_top10.csv
# - distinctive_noun_adj_by_category_top10.csv
infer_pos_slug <- function(path) {
  b <- basename(path)
  if (grepl("^distinctive_words_by_", b)) return("words")
  m <- str_match(b, "^distinctive_(.+?)_by_")
  if (!is.na(m[1,2]) && nchar(m[1,2]) > 0) return(m[1,2])
  return("words")
}
pos_title_from_slug <- function(slug) {
  if (is.null(slug) || is.na(slug) || slug == "" || slug == "words") return("Words")

  # Map slug tokens -> nice, pluralized labels
  # (supports only the POS you said you'll use)
  pos_map <- c(
    noun  = "Nouns",
    adj   = "Adjectives",
    verb  = "Verbs",
    propn = "Proper Nouns",
    pron  = "Pronouns"
  )

  parts <- unlist(strsplit(slug, "_", fixed = TRUE))
  # normalize to lowercase for safety
  parts <- tolower(parts)

  labels <- vapply(parts, function(p) {
    if (p %in% names(pos_map)) pos_map[[p]] else p
  }, FUN.VALUE = character(1))

  # De-dup (optional safety)
  labels <- unique(labels)

  if (length(labels) == 1) {
    return(labels[[1]])
  } else if (length(labels) == 2) {
    return(paste0(labels[[1]], " and ", labels[[2]]))
  } else {
    return(paste0(paste(labels[1:(length(labels)-1)], collapse = ", "), ", and ", labels[[length(labels)]]))
  }
}


# ══════════════════════════════════════════════════════════════════════════════
# 1) By Category
# ══════════════════════════════════════════════════════════════════════════════
message("\n[1/5] Creating visualizations for: By Category")

cat_csv <- list.files(
  file.path(INPUT_DIR, "by_category"),
  pattern = "^distinctive_.+_by_category_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(cat_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(cat_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  cat_data <- read_csv(cat_csv[1], show_col_types = FALSE) %>%
    mutate(
      category_label = format_category(category),
      category_label_wrapped = sapply(category_label, format_category_label)
    )

  top_n <- infer_top_n(cat_data, c("category"))

  if (!CSV_ONLY) {
    categories <- unique(cat_data$category_label_wrapped)
    n_cats <- length(categories)
    n_cols <- 4
    n_rows <- ceiling(n_cats / n_cols)

    # FIX: put words on y and use scale_y_reordered(); no coord_flip()
    # Use neutral light blue color
    p <- ggplot(cat_data, aes(
      x = G2,
      y = reorder_within(feature, G2, category_label_wrapped)
    )) +
      geom_col(fill = "#87CEEB", show.legend = FALSE) +
      scale_y_reordered() +
      facet_wrap(~ category_label_wrapped, scales = "free_y", ncol = n_cols) +
      labs(
        title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Category"),
        x = NULL,
        y = NULL
      ) +
      theme_minimal(base_size = 10) +
      theme(
        strip.text = element_text(face = "bold", size = 9, lineheight = 0.9),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.text.y = element_text(size = 8),
        axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank()
      )

    png_name <- paste0("distinctive_", POS_SLUG, "_by_category_top", top_n, ".png")
    png_path <- file.path(INPUT_DIR, "by_category", png_name)
    save_plot_auto_height(p, png_path, top_n, n_rows, width = 12)
    message("  → Saved: ", basename(png_path))
  } else {
    message("  → Skipping PNG generation (CSV only mode)")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 2) By Decade
# ══════════════════════════════════════════════════════════════════════════════
message("\n[2/5] Creating visualizations for: By Decade")

dec_csv <- list.files(
  file.path(INPUT_DIR, "by_decade"),
  pattern = "^distinctive_.+_by_decade_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(dec_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(dec_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  dec_data <- read_csv(dec_csv[1], show_col_types = FALSE) %>%
    mutate(decade_label = format_decade(decade))

  top_n <- infer_top_n(dec_data, c("decade"))

  if (!CSV_ONLY) {
    decades <- unique(dec_data$decade_label)
    n_decades <- length(decades)
    n_cols <- 5
    n_rows <- ceiling(n_decades / n_cols)

    # Use neutral light blue color
    p <- ggplot(dec_data, aes(
      x = G2,
      y = reorder_within(feature, G2, decade_label)
    )) +
      geom_col(fill = "#87CEEB", show.legend = FALSE) +
      scale_y_reordered() +
      facet_wrap(~ decade_label, scales = "free_y", ncol = n_cols) +
      labs(
        title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Decade"),
        x = NULL,
        y = NULL
      ) +
      theme_minimal(base_size = 10) +
      theme(
        strip.text = element_text(face = "bold", size = 9),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.text.y = element_text(size = 8),
        axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank()
      )

    png_name <- paste0("distinctive_", POS_SLUG, "_by_decade_top", top_n, ".png")
    png_path <- file.path(INPUT_DIR, "by_decade", png_name)
    save_plot_auto_height(p, png_path, top_n, n_rows, width = 15)
  } else {
    message("  → Skipping PNG generation (CSV only mode)")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 3) By Decade and Category - SEPARATE PLOT PER CATEGORY
# ══════════════════════════════════════════════════════════════════════════════
message("\n[3/5] Creating visualizations for: By Decade and Category")

dc_csv <- list.files(
  file.path(INPUT_DIR, "by_decade_and_category"),
  pattern = "^distinctive_.+_by_decade_category_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(dc_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(dc_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  dc_data <- read_csv(dc_csv[1], show_col_types = FALSE) %>%
    mutate(
      category_label = format_category(category),
      decade_label = format_decade(decade)
    )

  top_n <- infer_top_n(dc_data, c("decade", "category"))

  categories <- sort(unique(dc_data$category))

  for (cat in categories) {
    cat_data <- dc_data %>% filter(category == cat)
    if (nrow(cat_data) == 0) next

    if (!CSV_ONLY) {
      n_decades <- length(unique(cat_data$decade))
      cat_label <- format_category(cat)

      # FIX: words on y + scale_y_reordered(); no coord_flip()
      # Use neutral light blue color
      p <- ggplot(cat_data, aes(
        x = G2,
        y = reorder_within(feature, G2, decade_label)
      )) +
        geom_col(fill = "#87CEEB") +
        scale_y_reordered() +
        facet_wrap(~ decade_label, scales = "free_y", ncol = 5) +
        labs(
          title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Decade: ", cat_label),
          x = NULL,
          y = NULL
        ) +
        theme_minimal(base_size = 10) +
        theme(
          strip.text = element_text(face = "bold", size = 9),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
          axis.text.y = element_text(size = 7),
          axis.text.x = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.spacing = unit(0.3, "lines")
        )

      safe_cat <- str_replace_all(cat, "[^A-Za-z0-9]+", "_")
      png_name <- paste0("distinctive_", POS_SLUG, "_by_decade_", safe_cat, "_top", top_n, ".png")
      png_path <- file.path(INPUT_DIR, "by_decade_and_category", png_name)

      # Height based on number of decades (rows)
      n_rows <- ceiling(n_decades / 5)
      height <- max(8, n_rows * (top_n * 0.15 + 1.5))
      width <- 18

      ggsave(png_path, p, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
      message("  → Saved: ", basename(png_path))
    }
  }
  
  if (CSV_ONLY) {
    message("  → Skipped PNG generation (CSV only mode)")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 4) By Gender and Category (FIXED: separate Men/Women plots, single row labels, color-coded)
# ══════════════════════════════════════════════════════════════════════════════
message("\n[4/5] Creating visualizations for: By Gender and Category")

gc_csv <- list.files(
  file.path(INPUT_DIR, "by_gender_and_category"),
  pattern = "^distinctive_.+_by_gender_category_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(gc_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(gc_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  gc_data <- read_csv(gc_csv[1], show_col_types = FALSE) %>%
    mutate(
      gender_label = ifelse(gender == "F", "Women", "Men"),
      category_label = format_category(category)
    )

  top_n <- infer_top_n(gc_data, c("category", "gender"))
  
  # Use MIN_WORDS from args if provided, otherwise use top_n
  min_words_threshold <- if (!is.null(MIN_WORDS)) MIN_WORDS else top_n

  # Filter out categories where women have fewer than min_words_threshold words
  women_word_counts <- gc_data %>%
    filter(gender == "F") %>%
    group_by(category) %>%
    summarise(n_words = n(), .groups = "drop") %>%
    filter(n_words >= min_words_threshold)
  
  gc_data <- gc_data %>%
    filter(category %in% women_word_counts$category)

  top_n <- infer_top_n(gc_data, c("category", "gender"))

  if (!CSV_ONLY) {
    # Keep stable ordering of facet rows
    gc_data$category_label <- factor(gc_data$category_label, levels = sort(unique(gc_data$category_label)))
    gc_data$gender_label <- factor(gc_data$gender_label, levels = c("Men", "Women"))

    make_gender_plot_gc <- function(df, gender_name, show_strip) {
      df_g <- df %>% filter(gender_label == gender_name)
      
      # Color: Men = blue, Women = red
      bar_color <- ifelse(gender_name == "Men", "#3182bd", "#d62728")

      p <- ggplot(df_g, aes(
        x = G2,
        y = reorder_within(feature, G2, category_label)
      )) +
        geom_col(fill = bar_color) +
        scale_y_reordered() +
        facet_grid(category_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
        labs(title = gender_name, x = NULL, y = NULL) +
        theme_minimal(base_size = 11) +
        theme(
          strip.placement = "outside",
          strip.text.y = if (show_strip) {
            element_text(face = "bold", size = 9, angle = 90, hjust = 0.5)
          } else {
            element_blank()
          },
          strip.background = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          axis.text.y = element_text(size = 8),
          axis.text.x = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.spacing.y = unit(0.3, "lines"),
          panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.5)
        )
      
      return(p)
    }

    # Show strip only on left plot (Men)
    p_m <- make_gender_plot_gc(gc_data, "Men", show_strip = TRUE)
    p_w <- make_gender_plot_gc(gc_data, "Women", show_strip = FALSE)

    p <- (p_m | p_w) +
      plot_annotation(
        title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Gender and Category"),
        theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
      )

    png_name <- paste0("distinctive_", POS_SLUG, "_by_gender_category_top", top_n, ".png")
    png_path <- file.path(INPUT_DIR, "by_gender_and_category", png_name)

    n_categories <- length(unique(gc_data$category_label))
    width <- 16
    height <- max(20, n_categories * (top_n * 0.13 + 1.2))

    ggsave(png_path, p, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
    message("  → Saved: ", basename(png_name))
  } else {
    message("  → Skipping PNG generation (CSV only mode)")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 5) By Gender and Decade (NEW: separate Men/Women plots, single row labels, color-coded)
# ══════════════════════════════════════════════════════════════════════════════
message("\n[5/5] Creating visualizations for: By Gender and Decade")

gd_csv <- list.files(
  file.path(INPUT_DIR, "by_gender_and_decade"),
  pattern = "^distinctive_.+_by_gender_decade_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(gd_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(gd_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  gd_data <- read_csv(gd_csv[1], show_col_types = FALSE) %>%
    mutate(
      gender_label = ifelse(gender == "F", "Women", "Men"),
      decade_label = format_decade(decade)
    )

  top_n <- infer_top_n(gd_data, c("decade", "gender"))
  
  # Use MIN_WORDS from args if provided, otherwise use top_n
  min_words_threshold <- if (!is.null(MIN_WORDS)) MIN_WORDS else top_n

  # Filter out decades where women have fewer than min_words_threshold words
  women_word_counts <- gd_data %>%
    filter(gender == "F") %>%
    group_by(decade) %>%
    summarise(n_words = n(), .groups = "drop") %>%
    filter(n_words >= min_words_threshold)
  
  gd_data <- gd_data %>%
    filter(decade %in% women_word_counts$decade)

  top_n <- infer_top_n(gd_data, c("decade", "gender"))

  if (!CSV_ONLY) {
    # Keep stable ordering of facet rows
    gd_data$decade_label <- factor(gd_data$decade_label, levels = sort(unique(gd_data$decade_label)))
    gd_data$gender_label <- factor(gd_data$gender_label, levels = c("Men", "Women"))

    make_gender_plot_gd <- function(df, gender_name, show_strip) {
      df_g <- df %>% filter(gender_label == gender_name)
      
      # Color: Men = blue, Women = red
      bar_color <- ifelse(gender_name == "Men", "#3182bd", "#d62728")

      p <- ggplot(df_g, aes(
        x = G2,
        y = reorder_within(feature, G2, decade_label)
      )) +
        geom_col(fill = bar_color) +
        scale_y_reordered() +
        facet_grid(decade_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
        labs(title = gender_name, x = NULL, y = NULL) +
        theme_minimal(base_size = 11) +
        theme(
          strip.placement = "outside",
          strip.text.y = if (show_strip) {
            element_text(face = "bold", size = 9, angle = 90, hjust = 0.5)
          } else {
            element_blank()
          },
          strip.background = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          axis.text.y = element_text(size = 8),
          axis.text.x = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.spacing.y = unit(0.3, "lines"),
          panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.5)
        )
      
      return(p)
    }

    # Show strip only on left plot (Men)
    p_m <- make_gender_plot_gd(gd_data, "Men", show_strip = TRUE)
    p_w <- make_gender_plot_gd(gd_data, "Women", show_strip = FALSE)

    p <- (p_m | p_w) +
      plot_annotation(
        title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Gender and Decade"),
        theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
      )

    png_name <- paste0("distinctive_", POS_SLUG, "_by_gender_decade_top", top_n, ".png")
    png_path <- file.path(INPUT_DIR, "by_gender_and_decade", png_name)

    n_decades <- length(unique(gd_data$decade_label))
    width <- 16
    height <- max(20, n_decades * (top_n * 0.13 + 1.2))

    ggsave(png_path, p, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
    message("  → Saved: ", basename(png_name))
  } else {
    message("  → Skipping PNG generation (CSV only mode)")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 6) By Gender, Decade, and Category
#   FIXED: For by-decade and by-category views, build separate Men/Women plots so
#   each has its own word list. Single row labels (left side only). Color-coded bars.
# ══════════════════════════════════════════════════════════════════════════════
message("\n[6/6] Creating visualizations for: By Gender, Decade, and Category")

gdc_csv <- list.files(
  file.path(INPUT_DIR, "by_gender_decade_category"),
  pattern = "^distinctive_.+_by_gender_decade_category_top\\d+\\.csv$",
  full.names = TRUE
)

if (length(gdc_csv) == 0) {
  message("  → No CSV found, skipping")
} else {
  POS_SLUG <- infer_pos_slug(gdc_csv[1])
  POS_TITLE <- pos_title_from_slug(POS_SLUG)

  gdc_data <- read_csv(gdc_csv[1], show_col_types = FALSE) %>%
    mutate(
      gender_label = ifelse(gender == "F", "Women", "Men"),
      category_label = format_category(category),
      decade_label = format_decade(decade)
    )

  top_n <- infer_top_n(gdc_data, c("category", "decade", "gender"))
  
  # Use MIN_WORDS from args if provided, otherwise use top_n
  min_words_threshold <- if (!is.null(MIN_WORDS)) MIN_WORDS else top_n

  by_decade_dir <- file.path(INPUT_DIR, "by_gender_decade_category", "by_decade")
  by_category_dir <- file.path(INPUT_DIR, "by_gender_decade_category", "by_category")
  dir.create(by_decade_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(by_category_dir, recursive = TRUE, showWarnings = FALSE)

  # ── View 1: By Decade (one plot per decade) ───────────────────────────────
  message("\n  Creating by-decade visualizations...")

  decades <- sort(unique(gdc_data$decade))

  make_gender_plot_gdc_by_decade <- function(df, gender_name, show_strip) {
    df_g <- df %>% filter(gender_label == gender_name)
    
    # Color: Men = blue, Women = red
    bar_color <- ifelse(gender_name == "Men", "#3182bd", "#d62728")

    ggplot(df_g, aes(
      x = G2,
      y = reorder_within(feature, G2, category_label)
    )) +
      geom_col(fill = bar_color) +
      scale_y_reordered() +
      facet_grid(category_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
      labs(title = gender_name, x = NULL, y = NULL) +
      theme_minimal(base_size = 11) +
      theme(
        strip.placement = "outside",
        strip.text.y = if (show_strip) {
          element_text(face = "bold", size = 9, angle = 90, hjust = 0.5)
        } else {
          element_blank()
        },
        strip.background = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        axis.text.y = element_text(size = 8),
        axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.spacing.y = unit(0.3, "lines"),
        panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.5)
      )
  }

  for (dec in decades) {
    dec_data <- gdc_data %>% filter(decade == dec)
    if (nrow(dec_data) == 0) next

    # Filter out categories where women have fewer than min_words_threshold words in this decade
    women_word_counts <- dec_data %>%
      filter(gender == "F") %>%
      group_by(category) %>%
      summarise(n_words = n(), .groups = "drop") %>%
      filter(n_words >= min_words_threshold)
    
    dec_data <- dec_data %>%
      filter(category %in% women_word_counts$category)
    
    if (nrow(dec_data) == 0) next

    dec_label <- format_decade(dec)

    # Save decade-specific CSV (POS-aware filename)
    csv_path <- file.path(by_decade_dir, paste0("distinctive_", POS_SLUG, "_", dec, "s_top", top_n, ".csv"))
    write_csv(dec_data, csv_path)

    if (!CSV_ONLY) {
      # Stable row order
      dec_data$category_label <- factor(dec_data$category_label, levels = sort(unique(dec_data$category_label)))

      # Show strip only on left plot (Men)
      p_m <- make_gender_plot_gdc_by_decade(dec_data, "Men", show_strip = TRUE)
      p_w <- make_gender_plot_gdc_by_decade(dec_data, "Women", show_strip = FALSE)

      p <- (p_m | p_w) +
        plot_annotation(
          title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Gender and Category (", dec_label, ")"),
          theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
        )

      png_path <- file.path(by_decade_dir, paste0("distinctive_", POS_SLUG, "_", dec, "s_top", top_n, ".png"))

      n_cats_in_decade <- length(unique(dec_data$category_label))
      width <- 16
      height <- max(15, n_cats_in_decade * (top_n * 0.13 + 1.2))

      ggsave(png_path, p, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
      message("    → Saved: ", basename(png_path))
    } else {
      message("    → Saved CSV: ", basename(csv_path))
    }
  }
  
  if (CSV_ONLY) {
    message("  → Skipped PNG generation (CSV only mode)")
  }

  # ── View 2: By Category (one plot per category) ───────────────────────────
  message("\n  Creating by-category visualizations...")

  categories <- sort(unique(gdc_data$category))

  make_gender_plot_gdc_by_category <- function(df, gender_name, show_strip) {
    df_g <- df %>% filter(gender_label == gender_name)
    
    # Color: Men = blue, Women = red
    bar_color <- ifelse(gender_name == "Men", "#3182bd", "#d62728")

    ggplot(df_g, aes(
      x = G2,
      y = reorder_within(feature, G2, decade_label)
    )) +
      geom_col(fill = bar_color) +
      scale_y_reordered() +
      facet_grid(decade_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
      labs(title = gender_name, x = NULL, y = NULL) +
      theme_minimal(base_size = 11) +
      theme(
        strip.placement = "outside",
        strip.text.y = if (show_strip) {
          element_text(face = "bold", size = 9, angle = 90, hjust = 0.5)
        } else {
          element_blank()
        },
        strip.background = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        axis.text.y = element_text(size = 8),
        axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.spacing.y = unit(0.3, "lines"),
        panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.5)
      )
  }

  for (cat in categories) {
    cat_data <- gdc_data %>% filter(category == cat)
    if (nrow(cat_data) == 0) next

    # Filter out decades where women have fewer than min_words_threshold words in this category
    women_word_counts <- cat_data %>%
      filter(gender == "F") %>%
      group_by(decade) %>%
      summarise(n_words = n(), .groups = "drop") %>%
      filter(n_words >= min_words_threshold)
    
    cat_data <- cat_data %>%
      filter(decade %in% women_word_counts$decade)
    
    if (nrow(cat_data) == 0) next

    safe_cat <- str_replace_all(cat, "[^A-Za-z0-9]+", "_")
    cat_label <- format_category(cat)

    # Save category-specific CSV (POS-aware filename)
    csv_path <- file.path(by_category_dir, paste0("distinctive_", POS_SLUG, "_", safe_cat, "_top", top_n, ".csv"))
    write_csv(cat_data, csv_path)

    if (!CSV_ONLY) {
      # Stable row order
      cat_data$decade_label <- factor(cat_data$decade_label, levels = sort(unique(cat_data$decade_label)))

      # Show strip only on left plot (Men)
      p_m <- make_gender_plot_gdc_by_category(cat_data, "Men", show_strip = TRUE)
      p_w <- make_gender_plot_gdc_by_category(cat_data, "Women", show_strip = FALSE)

      p <- (p_m | p_w) +
        plot_annotation(
          title = paste0("Top ", top_n, " Distinctive ", POS_TITLE, " by Gender and Decade: ", cat_label),
          theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
        )

      png_path <- file.path(by_category_dir, paste0("distinctive_", POS_SLUG, "_", safe_cat, "_top", top_n, ".png"))

      n_decades <- length(unique(cat_data$decade))
      width <- 16
      height <- max(20, n_decades * (top_n * 0.13 + 1.2))

      ggsave(png_path, p, width = width, height = height, dpi = 300, bg = "white", limitsize = FALSE)
      message("    → Saved: ", basename(png_path))
    } else {
      message("    → Saved CSV: ", basename(csv_path))
    }
  }
  
  if (CSV_ONLY) {
    message("  → Skipped PNG generation (CSV only mode)")
  }
}

message("\n═══════════════════════════════════════════════════════════════")
message("COMPLETE! All visualizations saved to: ", INPUT_DIR)
message("═══════════════════════════════════════════════════════════════\n")
