# Lab 1: Tokenization for Hindi and Marathi Text

## Overview
This lab implements a comprehensive tokenization system for multilingual text processing, specifically designed for Indian languages (Hindi and Marathi) with support for English and various text patterns.

## Files
- `tokenizer.ipynb` - Main Jupyter notebook containing tokenization implementation
- `tokenized_hindi_sentences.json` - Output: Hindi text segmented into sentences
- `tokenized_hindi_tokens.json` - Output: Hindi text tokenized into individual words
- `tokenized_marathi_sentences.json` - Output: Marathi text segmented into sentences  
- `tokenized_marathi_tokens.json` - Output: Marathi text tokenized into individual words

## Features

### 1. Multi-language Support
- **Devanagari Script**: Hindi, Marathi, Sanskrit
- **Other Indian Scripts**: Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia
- **English**: Full alphabetic support
- **Arabic/Urdu**: Basic support

### 2. Advanced Pattern Recognition
- **URLs**: `http://`, `https://`, `www.` patterns
- **Email Addresses**: Complete email validation
- **Dates**: Multiple formats (DD/MM/YYYY, DD-MM-YYYY, etc.)
- **Numbers**: Integers and decimal numbers
- **Punctuation**: Comprehensive punctuation handling including Devanagari punctuation (।, ॥)

### 3. Tokenization Functions

#### Sentence Tokenizer
```python
def sentence_tokenizer(text)
```
- Splits text into sentences using multiple delimiters
- Handles Devanagari sentence endings (।, ॥)
- Supports English punctuation (., !, ?)

#### Word Tokenizer  
```python
def word_tokenizer(sentence)
```
- Regex-based tokenization with priority patterns
- Handles conjuncts and diacritics in Indian scripts
- Preserves structured data (URLs, emails, dates)

## Data Processing Pipeline

1. **Data Loading**: Uses HuggingFace `ai4bharat/IndicCorpV2` dataset
2. **Sentence Segmentation**: Text → Sentences
3. **Word Tokenization**: Sentences → Tokens
4. **Output Generation**: 
   - Sentence-level JSON files
   - Token-level JSON files
5. **Statistics Calculation**: Comprehensive corpus analysis

## Usage

### Running the Tokenizer
```python
# Load data
texts = load_data("hin_Deva", 50)  # Hindi
texts = load_data("mar_Deva", 25)  # Marathi

# Tokenize
tokenized_data = tokenize_texts(texts)

# Save results
save_tokenized_data(tokenized_data, "output_sentences.json")
save_tokenized_tokens(tokenized_data, "output_tokens.json")
```

### Statistics Generated
- Total paragraphs, sentences, and words
- Unique vocabulary size
- Average sentence length
- Type-token ratio
- Sample token display

## Output Format

### Sentence Files
```json
[
  [
    "यह एक अच्छा दिन है।",
    "भारत में शिक्षा का स्तर बढ़ रहा है।"
  ]
]
```

### Token Files
```json
[
  [
    ["यह", "एक", "अच्छा", "दिन", "है", "।"],
    ["भारत", "में", "शिक्षा", "का", "स्तर", "बढ़", "रहा", "है", "।"]
  ]
]
```

## Key Implementation Details

- **Memory-efficient**: Streaming approach for large datasets
- **Robust**: Error handling for malformed input
- **Extensible**: Easy to add new language scripts
- **Unicode-compliant**: Full Unicode support for all scripts
- **Statistical**: Built-in corpus analysis tools

## Requirements
- `datasets` (HuggingFace)
- `re` (regex)
- `json`
- `tqdm` (progress bars)

## Applications
- Preprocessing for NLP models
- Corpus linguistics research
- Multilingual text analysis
- Information extraction from Indian language texts