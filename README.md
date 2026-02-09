# Feminized Language of Democracy

## Collect the United States Congressional Records
Before running any scripts in this repository, you must collect the daily Congressional Record data:
- Required Repository: [uscongress-data](https://github.com/stephbuon/uscongress-data).
- Follow the instructions in that repository to generate: `congress_data_daily_by_speaker_with_metadata.csv`.


## Vision-LLM Text Transcription

Install ollama:

```
curl -fsSL https://ollama.com/install.sh | sh
```

Donwload Meta's Llama 3 11b vision model: 

```
ollama pull llama3.2-vision
```

Create an instance of the model wih custom parameters and a system prompt:  
```
ollama create llama3.2-vision-ocr -f ~/Repos/text-mine-congress/ocr/ocr.Modelfile
```

Run the model by pointing it to an image file and providing it with a prompt. Note: a system prompt may not be enough to generate wanted results.

```
ollama run llama3.2-vision-ocr "~/Repos/text-mine-congress/ocr/debug_images/image_a.jpg" "You are a high quality OCR tool. Provide me with a word-for-word transcription of this text. Do not repeat phrases and stop at the final line."
```

## Step 2: Process the Data using `pipeline.R`

`pipeline.R` calls these scripts: 

#### `create_decade_subset.R`
- Input: `congress_data_daily_by_speaker_with_metadata.csv`.
- Output: decade subsets stored in `.RData` format.

#### `parse_congress.R`
- Input: the `.RData` decade subsets. 
- Output: Two chunked `.parquet` files per decade (original + POS-parsed), each with `doc_id` and `chunk_id` fields (that together make a unique ID).

#### `bind_chunks.R`
- Input: The combined .parquet files by decade (original + POS-parsed).
- Output: Separate decade subsets as .parquet files for each gender (men and women).

#### `subset_by_gender.R`
- Input: The bound chunks as `.parquet` files.
- Output: Decade subsets for each gender

#### `remove_stopwords.R`
- Input: The decade `.parquet` files for each gender
- Output: A cleaned version of the decade `.parquet` files without stop words. 

#### `subset_by_category.R`
- Input:
  - Gender-separated decade .parquet files.
  - congress_controlled_vocab.csv file containing keyword categories and patterns.
- Output:
  - CSV debugger files containing matched documents with keyword counts.
  - `.parquet` files filtered by keyword categories for each decade, separated by gender.

## Step 3: Analyze the Data

#### `tf_idf.R`
- Input: the combined gender .parquet files and runs TF-IDF or log-likelihood
- Output: Visualizations
