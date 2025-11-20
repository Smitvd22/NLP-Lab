# Lab 3: Trie-based Stemming and Frequency Analysis

## Overview
This lab implements advanced text processing techniques using trie data structures for morphological analysis and frequency-based stemming, along with statistical analysis of token distributions.

## Files
- `trie_stemming.py` - Main trie-based stemming implementation
- `frequency_analysis.py` - Statistical frequency analysis and visualization
- `prefix_out.txt` - Prefix-based stemming results
- `suffix_out.txt` - Suffix-based stemming results  
- `trie_q1_output.txt` - Final combined stemming output with winner selection

## Part 1: Trie-based Stemming (`trie_stemming.py`)

### Core Concept
Uses trie (prefix tree) data structures to identify optimal morpheme boundaries in words by analyzing branching patterns and frequency distributions.

### Algorithm Components

#### 1. Trie Node Structure
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Character transitions
        self.cnt = 0       # Frequency count
```

#### 2. Scoring Function
**Branching Score Formula:**
```
score = fraction × branching_factor
where:
- fraction = 1.0 - (max_child_freq / total_freq)  
- branching_factor = number of child nodes
```

#### 3. Split Decision Criteria
- **Minimum Branching**: `BRANCH_THRESHOLD = 15`
- **Score-based**: Higher scores indicate better split points
- **Support-based**: Frequency-weighted decisions

### Morphological Analysis

#### Prefix Analysis
**Common Prefixes Detected:**
- `un-`, `re-`, `in-`, `im-`, `dis-`, `en-`, `non-`
- `over-`, `mis-`, `sub-`, `pre-`, `inter-`, `fore-`
- `de-`, `trans-`, `super-`, `semi-`, `anti-`, `mid-`, `under-`

#### Suffix Analysis  
**Common Suffixes Detected:**
- **Plurals**: `s`, `es`, `ies` → `y`
- **Verb forms**: `ing`, `ed`, `er`, `est`
- **Nominalizations**: `ment`, `ness`, `tion`
- **Adjective forms**: `able`, `ful`, `al`, `ous`, `ly`

### Output Format
```
word=stem+affix [score=X.XXXX, freq=N, prob=X.XXXX]
uncommon=un+common [freq=45, prob=0.0123, common_prefix]
running=runn+ing [score=2.1234, freq=12, prob=0.0034]
```

### Winner Selection
System compares prefix vs suffix approaches:
1. **Count Comparison**: Number of successful splits
2. **Score Sum**: Total splitting confidence  
3. **Winner Selection**: Best performing approach
4. **Final Output**: Winner's results copied to `trie_q1_output.txt`

## Part 2: Frequency Analysis (`frequency_analysis.py`)

### Purpose
Analyzes token frequency distributions from Hindi tokenized data, implements stop word detection, and generates statistical visualizations.

### Features

#### 1. Data Loading
- Loads from `Lab1/tokenized_hindi_tokens.json`
- Flattens nested JSON structure
- Handles various data formats gracefully

#### 2. Frequency Distribution
```python
def frequency_distribution(tokens):
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq
```

#### 3. Visualization
- **Font Support**: Devanagari-compatible fonts (Nirmala UI)
- **Top-N Plots**: Configurable frequency plots
- **Statistical Plots**: Before/after stop word removal

#### 4. Stop Word Detection
**Threshold-based Approach:**
```python
def find_stop_words(freq, threshold):
    return {word for word, count in freq.items() if count >= threshold}
```

**Multiple Thresholds:** 25, 50, 100 for comparative analysis

### Visualization Features
- **Bar Charts**: Top 100 most frequent words
- **Unicode Support**: Full Devanagari text rendering
- **Rotation**: 90-degree x-axis labels for readability
- **Comparative Analysis**: Before/after filtering

## Algorithm Details

### Trie Construction
1. **Word Insertion**: Characters inserted sequentially
2. **Frequency Tracking**: Each node maintains occurrence count
3. **Branching Analysis**: Child node distribution analysis

### Best Split Detection
```python
def best_split(root, word):
    # Forward trie analysis for prefixes
    # Returns: (split_index, score, support_frequency)

def best_split_suffix(root, word):  
    # Reverse trie analysis for suffixes
    # Returns: (split_index, score, support_frequency)
```

### Scoring Metrics
- **Fraction**: Measures distribution uniformity
- **Branching**: Counts continuation options
- **Support**: Frequency-based confidence
- **Threshold Filtering**: Minimum branching requirement

## Usage

### Running Stemming Analysis
```bash
python trie_stemming.py
```

**Process:**
1. Loads Brown corpus nouns
2. Builds prefix and suffix tries
3. Analyzes each word for optimal splits
4. Compares approaches and selects winner
5. Generates comprehensive output files

### Running Frequency Analysis
```bash
python frequency_analysis.py
```

**Process:**  
1. Loads Hindi tokenized data
2. Computes frequency distributions
3. Generates visualization plots
4. Tests multiple stop word thresholds
5. Shows comparative statistics

## Key Features

### Advanced Stemming
- **Dual Approach**: Both prefix and suffix analysis
- **Probabilistic Scoring**: Frequency-weighted decisions
- **Common Pattern Recognition**: Linguistic rule integration
- **Comparative Evaluation**: Automatic best approach selection

### Statistical Analysis
- **Multilingual Support**: Hindi/Devanagari processing
- **Robust Visualization**: Unicode-safe plotting
- **Threshold Analysis**: Multiple cutoff comparisons
- **Corpus Statistics**: Comprehensive metrics

### Performance Optimizations
- **Memory Efficiency**: Streamlined trie structures
- **Processing Speed**: Optimized splitting algorithms
- **Error Handling**: Graceful failure recovery
- **Scalability**: Large corpus support

## Applications
- **Morphological Analysis**: Word structure identification
- **Information Retrieval**: Document preprocessing
- **Corpus Linguistics**: Statistical text analysis  
- **Language Modeling**: Vocabulary optimization
- **Search Systems**: Query normalization

## Requirements
- `collections` (data structures)
- `defaultdict` (efficient dictionaries)
- `matplotlib` (visualization)
- `json` (data loading)
- `os` (file operations)

## Educational Value
- Demonstrates trie data structure applications
- Shows frequency-based linguistic analysis
- Illustrates morphological processing techniques
- Provides practical stemming algorithms
- Combines statistical and rule-based approaches