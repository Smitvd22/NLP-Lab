# Lab 4: N-gram Language Models

## Overview
This lab implements comprehensive n-gram language models (unigram, bigram, trigram, quadragram) for Hindi and Marathi languages using streaming algorithms and statistical modeling techniques.

## Files
- `LanguageModels.py` - Main n-gram model implementation
- `application.py` - Application interface for language models
- `smoothing.py` - Smoothing techniques implementation
- `news_articles.txt` - Input corpus text
- **Output TSV files:**
  - `hindi_unigrams.tsv`, `hindi_bigrams.tsv`, `hindi_trigrams.tsv`, `hindi_quadragrams.tsv`
  - `marathi_unigrams.tsv`, `marathi_bigrams.tsv`, `marathi_trigrams.tsv`, `marathi_quadragrams.tsv`
- `sentence_probabilities_from_smoothed.json` - Smoothed probability calculations
- `smoothing_results/` - Directory containing smoothing analysis results

## Core Implementation

### N-gram Model Architecture

#### 1. Streaming Token Processing
```python
def stream_tokens_from_json(tokens_list):
    """Memory-efficient token streaming"""
    for tok in tokens_list:
        if tok.strip():
            yield tok.strip()
```

#### 2. Rolling Window Algorithm  
- **Memory-Efficient**: Uses `deque` with `maxlen=MAX_N-1`
- **Real-time Processing**: Processes tokens one at a time
- **Multiple N-grams**: Generates 1-4 grams simultaneously

#### 3. Probability Calculations

**Unigram Probability (Maximum Likelihood):**
```
P(w) = Count(w) / Total_Tokens
```

**Conditional Probability (Higher N-grams):**
```
P(w_n | w_1...w_{n-1}) = Count(w_1...w_n) / Count(w_1...w_{n-1})
```

### Output Format

#### Unigrams TSV
```
token    count    p
यह       45       0.00234567
भारत     32       0.00167890
```

#### Higher N-grams TSV
```
w1       w2       w3       w4       count    p_cond
यह       एक       अच्छा    दिन       5        0.23456789
```

## Language Processing Pipeline

### 1. Data Loading
- Sources: `Lab1/tokenized_hindi_tokens.json`, `Lab1/tokenized_marathi_tokens.json`
- **Flattening**: Converts nested JSON to flat token list
- **Cleaning**: Removes empty tokens and normalizes

### 2. Statistical Modeling
```python
def process_language(language_name, tokens_list):
    # Streaming n-gram generation for orders 1-4
    # Vocabulary tracking and frequency counting
    # Probability computation and file output
```

### 3. Memory Optimization
- **Streaming Architecture**: No need to load entire corpus
- **Rolling Windows**: Fixed memory footprint regardless of corpus size
- **Incremental Processing**: Real-time statistics updates

## Advanced Features

### 1. Multi-order Processing
- **Simultaneous N-grams**: Generates 1-4 grams in single pass
- **Context Preservation**: Maintains proper n-gram histories
- **Efficient Storage**: Optimized data structures

### 2. Vocabulary Management
```python
# Real-time vocabulary building
vocab = set()
for tok in stream_tokens_from_json(tokens_list):
    vocab.add(tok)
    # Process n-grams using rolling window
```

### 3. Statistical Output
- **Corpus Statistics**: Token counts, vocabulary sizes
- **N-gram Statistics**: Unique n-gram counts per order
- **Top-N Display**: Most frequent n-grams per category

### Example Statistics Output:
```
Total tokens: 45,678
Vocabulary size: 8,234
Unique 1-grams: 8,234
Unique 2-grams: 23,456
Unique 3-grams: 34,567
Unique 4-grams: 41,234
```

## Smoothing Integration

### Smoothing Techniques
- **Add-k Smoothing**: Implemented in `smoothing.py`
- **Good-Turing**: Advanced frequency estimation
- **Interpolation**: Linear combination of n-gram orders
- **Backoff Models**: Hierarchical probability estimation

### Probability Calculations
- **Raw MLE**: Direct frequency ratios
- **Smoothed Probabilities**: Adjusted for unseen events
- **Sentence Probabilities**: Product of n-gram probabilities

## Usage

### Training Models
```python
# Load tokenized data
hindi_tokens = load_tokens('Lab1/tokenized_hindi_tokens.json')
marathi_tokens = load_tokens('Lab1/tokenized_marathi_tokens.json')

# Process each language
process_language("Hindi", hindi_tokens)
process_language("Marathi", marathi_tokens)
```

### Running Complete Pipeline
```bash
python LanguageModels.py
```

**Execution Flow:**
1. Load Hindi and Marathi tokenized data
2. Flatten nested token structures  
3. Process each language through streaming pipeline
4. Generate TSV output files for all n-gram orders
5. Display top-10 most frequent n-grams per order
6. Output comprehensive statistics

## Key Algorithms

### Rolling Window N-gram Generation
```python
window = deque(maxlen=MAX_N - 1)
for tok in stream_tokens_from_json(tokens_list):
    # Generate n-grams using current window + new token
    hist = list(window)
    for n in range(2, MAX_N + 1):
        if len(hist) >= n - 1:
            gram = tuple(hist[-(n-1):] + [tok])
            counts[n][gram] += 1
    window.append(tok)
```

### Probability Computation
```python
def conditional_prob(ngram, counts_n, counts_prev):
    history = ngram[:-1]  
    denom = counts_prev.get(history, 0)
    return counts_n.get(ngram, 0) / denom if denom else 0.0
```

## Performance Characteristics

### Memory Efficiency
- **Fixed Memory**: O(V + N×MAX_N) regardless of corpus size
- **Streaming**: No need to store entire corpus
- **Incremental**: Real-time processing capability

### Time Complexity
- **Linear Time**: O(T) where T = total tokens
- **Single Pass**: All n-gram orders in one iteration
- **Optimized Structures**: `defaultdict` for fast counting

### Scalability
- **Large Corpora**: Handles millions of tokens efficiently
- **Multiple Languages**: Parallel processing capability
- **Real-time**: Suitable for online applications

## Applications
- **Language Modeling**: Foundation for text generation
- **Machine Translation**: Translation probability estimation
- **Speech Recognition**: Language model scoring
- **Text Prediction**: Auto-completion systems
- **Information Retrieval**: Query understanding
- **Linguistic Analysis**: Statistical language patterns

## Requirements
- `collections` (defaultdict, deque)
- `json` (data loading)
- `pathlib` (file handling)
- `tqdm` (progress tracking)

## Educational Value
- Demonstrates streaming algorithms for NLP
- Shows practical n-gram implementation
- Illustrates memory-efficient text processing
- Provides foundation for advanced language modeling
- Combines theoretical concepts with practical implementation