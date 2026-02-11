# NLP Assignment 1 - Task 1: Tokenization

## Overview
This project implements three types of tokenizers (Whitespace, Regex, and BPE) for both English and Mongolian languages, as part of Task 1 of the NLP assignment.

## Files
- `tokenizers.py` - Implementation of all three tokenizers
- `utils.py` - Utility functions for corpus loading, cleaning, and partitioning
- `language_models.py` - N-gram language model and smoothing implementation
- `run_complete_assignment.py` - single script to generate all results/models
- `README.md`
- `report.pdf` 

## Corpus Statistics
- **English corpus**: 1,000,001 lines (~168 MB)
- **Mongolian corpus**: 1,000,001 lines (~266 MB)

## Task 1.1: Corpus Cleaning and Partitioning

### Data Partitioning Ratios
The corpus is partitioned into three sets with the following ratios:
- **Training set**: 80% (0.8)
- **Validation set**: 10% (0.1)
- **Testing set**: 10% (0.1)

**Rationale**: The 80/10/10 split is a standard practice in machine learning that provides:
- Sufficient training data for learning patterns
- Adequate validation data for hyperparameter tuning
- Enough test data for reliable performance evaluation

**Implementation**: The `partition_data()` function in `utils.py` performs line-level splitting (to avoid cutting sentences) with a fixed random seed (42) for reproducibility.

### Corpus Cleaning
The `clean_text()` function in `utils.py` performs the following cleaning operations:

1. **Remove control characters**: Eliminates null bytes and control characters (except newlines and tabs)
2. **Remove zero-width Unicode characters**: Strips invisible Unicode artifacts (U+200B-U+206F)
3. **Normalize spacing**: Reduces multiple spaces to single spaces
4. **Normalize newlines**: Limits consecutive newlines to maximum of 2
5. **Trim whitespace**: Removes leading/trailing spaces on each line
6. **Strip empty lines**: Removes empty lines at start/end of corpus

**Rationale**: These cleaning steps remove common artifacts from web-scraped text while preserving the actual content and structure.

## Task 1.2: Tokenizer Implementations

### 1. Whitespace Tokenizer
**Approach**: Splits text on whitespace, then uses regex `\w+|[^\w\s]` to separate alphanumeric characters from punctuation.

**Simplifying Assumptions**:
- Whitespace is the primary word boundary
- All special characters should be separate tokens
- No language-specific rules needed

**Strengths**: Simple, fast, language-agnostic
**Weaknesses**: Poor handling of contractions, URLs, and numbers with punctuation

### 2. Regex Tokenizer
**Approach**: Uses language-specific regex patterns to identify tokens.

**English Pattern**: `\w+(?:'\w+)?|[^\w\s]`
- Captures words with optional apostrophe contractions (e.g., "can't", "it's")
- Separates punctuation as individual tokens

**Mongolian Pattern**: `[\u0400-\u04FF\w]+|[^\w\s]`
- Captures Cyrillic characters (Unicode range U+0400 to U+04FF)
- Handles Mongolian-specific characters (ө, ү, etc.)
- Separates punctuation as individual tokens

### 3. BPE (Byte-Pair Encoding) Tokenizer
**Approach**: Starts with character-level tokens and iteratively merges the most frequent adjacent pairs.

**Algorithm**:
1. Initialize vocabulary with individual characters
2. Count all adjacent character pairs in the corpus
3. Merge the most frequent pair (if above minimum frequency threshold)
4. Repeat until vocabulary size is reached or no pairs meet frequency threshold
5. Use `</w>` as end-of-word marker to preserve word boundaries

**Parameters**:
- `vocab_size`: Maximum number of merge operations (default: 5000)
- `min_frequency`: Minimum frequency for a pair to be merged (default: 3)
- `unk_token`: Token for unknown characters (default: `<UNK>`)
---

## How to Run This Assignment

### Prerequisites
- Python 3.7+
- Standard library only (no external NLP libraries)
- English corpus file: `cc100_en.jsonl`
- Mongolian corpus file: `cc100_mn.jsonl`

### Step 1: Preprocess Data
Run the utility script to clean and partition the data. This creates the `data/` directory with train/val/test splits.

```bash
python utils.py
```

### Step 2: Run Complete Assignment
Run the main script to execute the entire pipeline (Tokenization analysis, Training, Evaluation, and Autocomplete).

```bash
python run_complete_assignment.py
```

This script will:
1. Load the preprocessed data.
2. Train all 6 tokenizers (3 English, 3 Mongolian).
3. Train 9 language models (3 tokenizers × 3 smoothing methods).
4. Evaluate perplexity on validation and test sets (Task 2.3).
5. Save all models to `models/` directory.



