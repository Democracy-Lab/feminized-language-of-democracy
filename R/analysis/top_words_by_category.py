from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple, List, Optional
import warnings
import requests
import unicodedata

import pandas as pd

# For stopwords
try:
    from nltk.corpus import stopwords as nltk_stopwords
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    raise ImportError("nltk is required. Install with: pip install nltk")


def safelabel(x: str) -> str:
    """Match the safelabel function from subset_by_category.py"""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))


def get_cli_opt(name: str, default: str | None = None) -> str | None:
    """Parse CLI arguments in format --name=value or --name value"""
    a = sys.argv[1:]
    # --name=value
    for arg in a:
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
    # --name value
    for i, arg in enumerate(a):
        if arg == f"--{name}" and i + 1 < len(a):
            return a[i + 1]
    return default


def load_congress_stopwords() -> set:
    """
    Load congress-specific stopwords from GitHub repository.
    Returns a set of stopwords.
    """
    url = "https://raw.githubusercontent.com/Democracy-Lab/feminized-language-of-democracy/main/analysis/congress_stopwords.csv"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse stopwords - one per line, no commas
        stopwords = set()
        for line in response.text.strip().split('\n'):
            word = line.strip()
            if word:  # Skip empty lines
                stopwords.add(word.lower())
        
        return stopwords
    except Exception as e:
        warnings.warn(f"Failed to load congress stopwords from GitHub: {e}")
        return set()


def initialize_stopwords() -> set:
    """
    Initialize combined stopword list from congress stopwords and NLTK English stopwords.
    Returns a set of stopwords.
    """
    # Download NLTK stopwords if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    # Get NLTK English stopwords
    nltk_stops = set(nltk_stopwords.words('english'))
    
    # Get congress stopwords
    congress_stops = load_congress_stopwords()
    
    # Combine both sets
    combined_stops = nltk_stops | congress_stops
    
    print(f"Loaded {len(nltk_stops)} NLTK stopwords")
    print(f"Loaded {len(congress_stops)} congress stopwords")
    print(f"Total unique stopwords: {len(combined_stops)}")
    
    return combined_stops


def clean_word(word: str, stopwords_set: set) -> Optional[str]:
    """
    Clean a word by removing punctuation, converting to lowercase, and filtering stopwords.
    Returns None if word should be filtered out.
    """
    word = unicodedata.normalize("NFKD", word)
    
    # Remove ALL punctuation using Unicode categories
    cleaned = ''.join(
        char for char in word 
        if not unicodedata.category(char).startswith('P')
    ).strip()
    
    # Convert to lowercase
    cleaned_lower = cleaned.lower()
    
    # Filter out stopwords
    if cleaned_lower in stopwords_set:
        return None
    
    # Filter out single-character tokens AND empty strings
    if len(cleaned_lower) <= 1:
        return None
    
    return cleaned_lower


def process_category_file_with_gender(args: Tuple[str, str, Path, set, int]) -> Tuple[str, str, Counter, dict]:
    """
    Process a single category-decade file and return word counts overall and by gender.
    
    OPTIMIZATION: Get top N*2 words first, THEN apply cleaning to reduce computation.
    
    Parameters
    ----------
    args : tuple
        (category, decade, file_path, stopwords_set, initial_top_n)
    
    Returns
    -------
    tuple
        (category, decade, overall_counter, gender_counters_dict)
        where gender_counters_dict is {'M': Counter(), 'F': Counter()}
    """
    category, decade, file_path, stopwords_set, initial_top_n = args
    
    # Try reading with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            if "token" not in df.columns:
                print(f"    Warning: 'token' column not found in {file_path.name}, skipping")
                return (category, decade, Counter(), {})
            
            # OPTIMIZATION: Get raw counts first (very fast - just Counter on raw tokens)
            raw_tokens = df["token"].astype(str).tolist()
            raw_counter = Counter(raw_tokens)
            
            # Get top N*2 most common raw tokens (before cleaning)
            # We take 2x to ensure we have enough after filtering
            top_raw = raw_counter.most_common(initial_top_n * 2)
            
            # NOW apply cleaning only to the top candidates
            cleaned_counter = Counter()
            for word, count in top_raw:
                cleaned = clean_word(word, stopwords_set)
                if cleaned is not None:
                    cleaned_counter[cleaned] += count
            
            overall_counter = cleaned_counter
            
            # Extract gender-specific counters if gender column exists
            gender_counters = {}
            if "gender" in df.columns:
                for gender in ['M', 'F']:
                    gender_mask = df["gender"] == gender
                    gender_raw_tokens = df[gender_mask]["token"].astype(str).tolist()
                    
                    # Same optimization: raw count first, then clean top N*2
                    gender_raw_counter = Counter(gender_raw_tokens)
                    top_gender_raw = gender_raw_counter.most_common(initial_top_n * 2)
                    
                    gender_cleaned_counter = Counter()
                    for word, count in top_gender_raw:
                        cleaned = clean_word(word, stopwords_set)
                        if cleaned is not None:
                            gender_cleaned_counter[cleaned] += count
                    
                    gender_counters[gender] = gender_cleaned_counter
            else:
                # If no gender column, create empty counters
                gender_counters = {'M': Counter(), 'F': Counter()}
            
            # Clear df to free memory
            del df
            
            m_count = sum(gender_counters.get('M', Counter()).values())
            f_count = sum(gender_counters.get('F', Counter()).values())
            total_count = sum(overall_counter.values())
            print(f"    Processed {file_path.name}: {total_count:,} tokens (M={m_count:,}, F={f_count:,})")
            return (category, decade, overall_counter, gender_counters)
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Attempt {attempt + 1} failed for {file_path.name}: {e}. Retrying...")
                import time
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"    Error reading {file_path.name} after {max_retries} attempts: {e}")
                # Try alternative: read as CSV if parquet fails
                try:
                    csv_path = file_path.with_suffix('.csv')
                    if csv_path.exists():
                        print(f"    Attempting to read CSV version: {csv_path.name}")
                        df = pd.read_csv(csv_path)
                        if "token" in df.columns:
                            raw_tokens = df["token"].astype(str).tolist()
                            raw_counter = Counter(raw_tokens)
                            top_raw = raw_counter.most_common(initial_top_n * 2)
                            
                            cleaned_counter = Counter()
                            for word, count in top_raw:
                                cleaned = clean_word(word, stopwords_set)
                                if cleaned is not None:
                                    cleaned_counter[cleaned] += count
                            
                            overall_counter = cleaned_counter
                            
                            gender_counters = {}
                            if "gender" in df.columns:
                                for gender in ['M', 'F']:
                                    gender_mask = df["gender"] == gender
                                    gender_raw_tokens = df[gender_mask]["token"].astype(str).tolist()
                                    gender_raw_counter = Counter(gender_raw_tokens)
                                    top_gender_raw = gender_raw_counter.most_common(initial_top_n * 2)
                                    
                                    gender_cleaned_counter = Counter()
                                    for word, count in top_gender_raw:
                                        cleaned = clean_word(word, stopwords_set)
                                        if cleaned is not None:
                                            gender_cleaned_counter[cleaned] += count
                                    
                                    gender_counters[gender] = gender_cleaned_counter
                            else:
                                gender_counters = {'M': Counter(), 'F': Counter()}
                            
                            del df
                            total_count = sum(overall_counter.values())
                            print(f"    Successfully read from CSV: {total_count:,} tokens")
                            return (category, decade, overall_counter, gender_counters)
                except:
                    pass
                return (category, decade, Counter(), {})


def extract_top_words(
    categories_dir: str | Path,
    output_dir: str | Path,
    thr: float = 0.03,
    pom: float = 0.85,
    top_n: int = 1000,
    workers: int = 1,
) -> None:
    """
    Extract top N most frequent words for multiple views:
    - by decade (all categories combined)
    - by category (all decades combined)
    - by category-decade
    - by decade-gender (all categories combined)
    - by category-decade-gender
    
    OPTIMIZED: Gets top N*2 raw words first, then applies cleaning to reduce computation.
    
    Parameters
    ----------
    categories_dir : str | Path
        Base directory containing the tfidf_norm_filtered_tokens_{tag} folder
    output_dir : str | Path
        Directory where output CSVs will be saved (subdirectories created for each view)
    thr : float
        Threshold value used in categorization (for path construction)
    pom : float
        Proportion of max value used in categorization (for path construction)
    top_n : int
        Number of top words to extract (default: 1000)
    workers : int
        Number of parallel workers (default: 1)
    """
    categories_dir = Path(categories_dir)
    output_dir = Path(output_dir)
    
    # Initialize stopwords
    print("Initializing stopwords...")
    stopwords_set = initialize_stopwords()
    print()
    
    # Create subdirectories for each view
    by_decade_dir = output_dir / "by_decade"
    by_category_dir = output_dir / "by_category"
    by_category_decade_dir = output_dir / "by_category_decade"
    by_decade_gender_dir = output_dir / "by_decade_gender"
    by_category_decade_gender_dir = output_dir / "by_category_decade_gender"
    
    for d in [by_decade_dir, by_category_dir, by_category_decade_dir, 
              by_decade_gender_dir, by_category_decade_gender_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    tag = f"thr{thr:.3f}_pom{pom:.2f}"
    filtered_tokens_dir = categories_dir / f"tfidf_norm_filtered_tokens_{tag}"
    
    if not filtered_tokens_dir.exists():
        raise FileNotFoundError(
            f"Filtered tokens directory not found: {filtered_tokens_dir}\n"
            f"Make sure subset_by_category.py has been run first."
        )
    
    # Find all filtered token parquet files
    token_files = list(filtered_tokens_dir.glob("*_filtered_tokens.parquet"))
    
    if not token_files:
        raise FileNotFoundError(
            f"No filtered token files found in: {filtered_tokens_dir}\n"
            f"Pattern expected: *_filtered_tokens.parquet"
        )
    
    print(f"Found {len(token_files)} filtered token files")
    print(f"Using {workers} worker(s)")
    print(f"Optimization: Processing top {top_n*2} raw tokens first, then cleaning")
    
    # Group files by category and prepare work items
    category_files = {}
    work_items: List[Tuple[str, str, Path, set, int]] = []
    
    for file_path in token_files:
        # Parse filename: {category}_{decade}_unit-debate_tfidf_norm_1k_filtered_tokens.parquet
        name = file_path.stem
        parts = name.split("_")
        
        # Find decade (4-digit number)
        decade = None
        category_parts = []
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                decade = part
                category_parts = parts[:i]
                break
        
        if decade is None:
            print(f"Warning: Could not parse decade from filename: {file_path.name}")
            continue
        
        category = "_".join(category_parts)
        
        if category not in category_files:
            category_files[category] = []
        category_files[category].append((decade, file_path))
        
        # Add to work items for parallel processing (now includes stopwords_set and initial_top_n)
        work_items.append((category, decade, file_path, stopwords_set, top_n))
    
    print(f"Processing {len(category_files)} categories")
    
    # Initialize all aggregation counters
    # View 1: by decade (all categories combined)
    decade_counters = {}
    
    # View 2: by category (all decades combined)
    category_counters = {}
    
    # View 3: by category-decade (saved immediately, no aggregation needed)
    category_decade_saved = set()
    
    # View 4: by decade-gender (all categories combined)
    decade_gender_counters = {}
    
    # View 5: by category-decade-gender (saved immediately, no aggregation needed)
    category_decade_gender_saved = set()
    
    # Process files in parallel - now also extracting gender info
    print("\nProcessing files in parallel and saving incrementally...")
    
    # Use imap_unordered to process results as they complete (streaming)
    # instead of waiting for all 192 files to finish
    if workers > 1:
        pool = Pool(processes=workers)
        results_iterator = pool.imap_unordered(process_category_file_with_gender, work_items, chunksize=1)
    else:
        results_iterator = (process_category_file_with_gender(item) for item in work_items)
    
    # STREAMING LOOP - process each result immediately as it arrives
    for i, (category, decade, overall_counter, gender_counters) in enumerate(results_iterator, 1):
        print(f"[{i}/{len(work_items)}] Aggregating and saving {category} {decade}...")
        
        # View 1: by decade
        if decade not in decade_counters:
            decade_counters[decade] = Counter()
        decade_counters[decade].update(overall_counter)
        
        # View 2: by category
        if category not in category_counters:
            category_counters[category] = Counter()
        category_counters[category].update(overall_counter)
        
        # View 3: by category-decade - SAVE IMMEDIATELY
        if overall_counter:
            top_words = overall_counter.most_common(top_n)
            df = pd.DataFrame(top_words, columns=["word", "count"])
            csv_path = by_category_decade_dir / f"top_{top_n}_words_{category}_{decade}.csv"
            df.to_csv(csv_path, index=False)
            category_decade_saved.add((category, decade))
            print(f"  ✓ Saved {category} {decade}: {csv_path.name}")
        
        # View 4 & 5: by decade-gender and category-decade-gender
        for gender, gender_counter in gender_counters.items():
            # View 4: by decade-gender
            key_dec_gen = (decade, gender)
            if key_dec_gen not in decade_gender_counters:
                decade_gender_counters[key_dec_gen] = Counter()
            decade_gender_counters[key_dec_gen].update(gender_counter)
            
            # View 5: by category-decade-gender - SAVE IMMEDIATELY
            if gender_counter:
                top_words = gender_counter.most_common(top_n)
                df = pd.DataFrame(top_words, columns=["word", "count"])
                csv_path = by_category_decade_gender_dir / f"top_{top_n}_words_{category}_{decade}_{gender}.csv"
                df.to_csv(csv_path, index=False)
                category_decade_gender_saved.add((category, decade, gender))
                print(f"  ✓ Saved {category} {decade} {gender}: {csv_path.name}")
    
    # Close the pool if we opened it
    if workers > 1:
        pool.close()
        pool.join()
        
        if gender_counter:
            top_words = gender_counter.most_common(top_n)
            df = pd.DataFrame(top_words, columns=["word", "count"])
            csv_path = by_category_decade_gender_dir / f"top_{top_n}_words_{category}_{decade}_{gender}.csv"
            df.to_csv(csv_path, index=False)
            category_decade_gender_saved.add((category, decade, gender))
            print(f"  ✓ Saved {category} {decade} {gender}: {csv_path.name}")
    
    # Save View 1: by decade (aggregated across categories)
    print("\nSaving by_decade results (aggregated across all categories)...")
    for decade, counter in decade_counters.items():
        if counter:
            top_words = counter.most_common(top_n)
            df = pd.DataFrame(top_words, columns=["word", "count"])
            csv_path = by_decade_dir / f"top_{top_n}_words_{decade}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved decade {decade}: {csv_path.name}")
    
    # Save View 2: by category (aggregated across decades)
    print("\nSaving by_category results (aggregated across all decades)...")
    for category, counter in category_counters.items():
        if counter:
            top_words = counter.most_common(top_n)
            df = pd.DataFrame(top_words, columns=["word", "count"])
            csv_path = by_category_dir / f"top_{top_n}_words_{category}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved category {category}: {csv_path.name}")
    
    # Save View 4: by decade-gender (aggregated across categories)
    print("\nSaving by_decade_gender results (aggregated across all categories)...")
    for (decade, gender), counter in decade_gender_counters.items():
        if counter:
            top_words = counter.most_common(top_n)
            df = pd.DataFrame(top_words, columns=["word", "count"])
            csv_path = by_decade_gender_dir / f"top_{top_n}_words_{decade}_{gender}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved decade {decade} gender {gender}: {csv_path.name}")
    
    print(f"\n✓ All top word files saved to: {output_dir}")
    print(f"  - by_decade: {len(decade_counters)} files")
    print(f"  - by_category: {len(category_counters)} files")
    print(f"  - by_category_decade: {len(category_decade_saved)} files")
    print(f"  - by_decade_gender: {len(decade_gender_counters)} files")
    print(f"  - by_category_decade_gender: {len(category_decade_gender_saved)} files")


if __name__ == "__main__":
    # Parse command-line arguments
    WORKERS = int(get_cli_opt("workers", "1"))
    
    # Paths - update these to match your setup
    CATEGORIES_DIR = Path("/local/scratch/group/guldigroup/climate_change/congress/text-mine-congress-python/code/data_01-2026/gender_analysis/categories")
    OUTPUT_DIR = Path("top_words_by_category_02-11-26")
    
    # These should match what you used in subset_by_category.py
    THR = 0.03
    POM = 0.85
    
    extract_top_words(
        categories_dir=CATEGORIES_DIR,
        output_dir=OUTPUT_DIR,
        thr=THR,
        pom=POM,
        top_n=1000,
        workers=WORKERS,
    )