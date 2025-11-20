# Lab 9: Subword Tokenization - BPE and WordPiece

## Overview
This lab implements two fundamental subword tokenization algorithms: Byte Pair Encoding (BPE) and WordPiece, designed to handle multilingual text including Hindi and Marathi from Assignment 1 corpus.

## Files
- `bpe.py` - Complete implementation of BPE and WordPiece algorithms

## Subword Tokenization Concepts

### Why Subword Tokenization?
1. **Out-of-Vocabulary Problem**: Handle unseen words gracefully
2. **Morphological Complexity**: Break words into meaningful components
3. **Multilingual Support**: Unified vocabulary across languages
4. **Efficient Representation**: Compact vocabulary with good coverage

### Applications
- Machine Translation systems (Google Translate)
- Language Models (BERT, GPT)
- Speech Recognition systems
- Information Retrieval systems

## Algorithm 1: Byte Pair Encoding (BPE)

### Theoretical Foundation
BPE iteratively merges the most frequent adjacent character pairs to build a vocabulary of subword units.

#### Algorithm Steps:
1. **Initialize**: Start with character-level vocabulary
2. **Count Pairs**: Find most frequent adjacent symbol pairs
3. **Merge**: Replace most frequent pair with new symbol
4. **Repeat**: Continue until desired vocabulary size
5. **Encode**: Apply learned merges to segment new text

### Mathematical Formulation
```
Vocab₀ = {all characters in corpus}
for i = 1 to num_merges:
    pair* = argmax_pair count(pair)
    Vocab_i = Vocab_{i-1} ∪ {merge(pair*)}
    replace all occurrences of pair* with merge(pair*)
```

### Implementation

#### 1. Core BPE Class
```python
class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = {}
        self.vocab = set()
```

#### 2. Enhanced Pre-tokenization
```python
def pre_tokenize(self, text):
    patterns = [
        r'[\u0900-\u097F]+',  # Devanagari (Hindi, Marathi)
        r'[a-zA-Z]+',         # English words
        r'\d+',               # Numbers
        r'[^\w\s]'            # Punctuation
    ]
    # Multilingual pattern matching with priority
```

#### 3. Statistical Analysis
```python
def get_stats(self, splits):
    """Count frequency of adjacent symbol pairs"""
    pairs = defaultdict(int)
    for word, freq in self.word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs
```

#### 4. Merge Operation
```python
def merge_symbols(self, pair, splits):
    """Merge most frequent pair in all words"""
    first, second = pair
    for word in splits:
        symbols = splits[word]
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1 and 
                symbols[i] == first and symbols[i + 1] == second):
                new_symbols.append(first + second)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        splits[word] = new_symbols
    return splits
```

### Training Process
```python
def train(self, corpus, num_merges=32000):
    # 1. Build word frequency table
    self.word_freqs = Counter(all_words)
    
    # 2. Initialize character-level splits
    self.splits = {word: list(word) for word in self.word_freqs}
    
    # 3. Build initial character vocabulary
    self.vocab = {char for word in self.word_freqs for char in word}
    
    # 4. Perform iterative merging
    for i in range(num_merges):
        pairs = self.get_stats(self.splits)
        if not pairs: break
        
        best_pair = max(pairs, key=pairs.get)
        self.splits = self.merge_symbols(best_pair, self.splits)
        self.merges[best_pair] = i
        self.vocab.add(''.join(best_pair))
```

## Algorithm 2: WordPiece

### Theoretical Foundation
WordPiece uses likelihood-based scoring to build vocabulary, preferring subwords that maximize the likelihood of the training corpus.

#### Key Differences from BPE:
- **Likelihood-based**: Uses statistical scoring instead of frequency
- **Greedy Selection**: Chooses pieces that best improve model likelihood
- **Continuation Markers**: Uses `##` prefix for non-initial pieces
- **Bottom-up Construction**: Builds from characters to longer pieces

### Mathematical Scoring
```
Likelihood(piece) = Count(words_containing_piece) / Length(piece)
```

### Implementation

#### 1. WordPiece Class Structure
```python
class WordPieceTokenizer:
    def __init__(self, vocab_size=32000, unk_token="[UNK]"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = set()
        self.word_freqs = {}
        self.subword_counts = defaultdict(int)
```

#### 2. Candidate Generation
```python
def get_word_pieces(self, word):
    """Generate all possible subword pieces"""
    pieces = []
    for i in range(len(word)):
        for j in range(i + 1, len(word) + 1):
            piece = word[i:j]
            if i > 0:
                piece = "##" + piece  # Continuation prefix
            pieces.append(piece)
    return pieces
```

#### 3. Likelihood Calculation
```python
def calculate_likelihood(self, piece):
    """Calculate likelihood score for WordPiece"""
    clean_piece = piece[2:] if piece.startswith("##") else piece
    
    count = 0
    for word, freq in self.word_freqs.items():
        if clean_piece in word:
            count += freq
    
    return count / len(clean_piece) if len(clean_piece) > 0 else 0
```

#### 4. Iterative Vocabulary Building
```python
def train(self, corpus, vocab_size=32000):
    # 1. Build word frequencies
    self.word_freqs = Counter(all_words)
    
    # 2. Initialize with characters and special tokens
    self.vocab = {self.unk_token}
    for word in self.word_freqs:
        for char in word:
            self.vocab.add(char)
    
    # 3. Generate candidate pieces
    all_candidates = set()
    for word in self.word_freqs:
        pieces = self.get_word_pieces(word)
        all_candidates.update(pieces)
    
    # 4. Iteratively add best pieces
    while len(self.vocab) < vocab_size and all_candidates:
        best_piece = max(candidates, key=self.calculate_likelihood)
        self.vocab.add(best_piece)
        all_candidates.remove(best_piece)
```

## Encoding Algorithms

### BPE Encoding
```python
def encode(self, text):
    words = self.pre_tokenize(text)
    encoded = []
    
    for word in words:
        if word in self.splits:
            encoded.extend(self.splits[word])
        else:
            # Apply learned merges to new words
            word_tokens = list(word)
            for (first, second), _ in sorted(self.merges.items(), key=lambda x: x[1]):
                # Apply merge operations in order
                word_tokens = self.apply_merge(word_tokens, first, second)
            encoded.extend(word_tokens)
    return encoded
```

### WordPiece Encoding
```python
def encode(self, text):
    words = self.pre_tokenize(text)
    tokens = []
    
    for word in words:
        word_tokens = []
        start = 0
        
        # Greedy longest-match algorithm
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Find longest matching subword
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                word_tokens.append(self.unk_token)
                start += 1
            else:
                word_tokens.append(cur_substr)
                start = start + len(cur_substr) - (2 if cur_substr.startswith("##") else 0)
        
        tokens.extend(word_tokens)
    return tokens
```

## Multilingual Support

### Enhanced Pre-tokenization Patterns
```python
patterns = [
    r'[\u0900-\u097F]+',  # Devanagari (Hindi, Marathi, Sanskrit)
    r'[\u0A80-\u0AFF]+',  # Gujarati
    r'[\u0980-\u09FF]+',  # Bengali, Assamese
    r'[\u0B80-\u0BFF]+',  # Tamil
    r'[\u0C00-\u0C7F]+',  # Telugu
    r'[\u0C80-\u0CFF]+',  # Kannada
    r'[\u0D00-\u0D7F]+',  # Malayalam
    r'[\u0A00-\u0A7F]+',  # Punjabi (Gurmukhi)
    r'[\u0B00-\u0B7F]+',  # Odia
    r'[a-zA-Z]+',         # English words
    r'\d+',               # Numbers
    r'[^\w\s]'            # Punctuation
]
```

### Data Loading from Assignment 1
```python
def load_assignment1_corpus():
    corpus_texts = []
    
    # Load Hindi sentences
    with open("../Lab1/tokenized_hindi_sentences.json", 'r') as f:
        hindi_data = json.load(f)
        for paragraph in hindi_data:
            for sentence in paragraph:
                if sentence.strip():
                    corpus_texts.append(sentence)
    
    # Load Marathi sentences  
    with open("../Lab1/tokenized_marathi_sentences.json", 'r') as f:
        marathi_data = json.load(f)
        for paragraph in marathi_data:
            for sentence in paragraph:
                if sentence.strip():
                    corpus_texts.append(sentence)
    
    return corpus_texts
```

## Usage and Execution

### Running the Implementation
```python
def main():
    # Load corpus from Assignment 1
    corpus = load_assignment1_corpus()
    
    # Training parameters
    NUM_MERGES = 32000
    VOCAB_SIZE = 32000
    
    # Train BPE Tokenizer
    bpe = BPETokenizer(vocab_size=VOCAB_SIZE)
    bpe.train(corpus, num_merges=NUM_MERGES)
    
    # Train WordPiece Tokenizer
    wordpiece = WordPieceTokenizer(vocab_size=VOCAB_SIZE)
    wordpiece.train(corpus, vocab_size=VOCAB_SIZE)
    
    # Test encoding
    test_texts = [
        "प्राकृतिक भाषा प्रसंस्करण",
        "मशीन लर्निंग एल्गोरिदम", 
        "संगणक विज्ञान",
        "natural language processing"
    ]
    
    for test_text in test_texts:
        bpe_tokens = bpe.encode(test_text)
        wp_tokens = wordpiece.encode(test_text)
        print(f"Text: {test_text}")
        print(f"BPE: {bpe_tokens}")
        print(f"WordPiece: {wp_tokens}")
```

### Execution
```bash
python bpe.py
```

**Process:**
1. Load Hindi and Marathi sentences from Lab 1
2. Train BPE tokenizer with 32,000 merge operations
3. Train WordPiece tokenizer with 32,000 vocabulary size
4. Test both tokenizers on sample texts
5. Display comparative results and statistics

## Output Analysis

### Training Statistics
```
Training with 32000 merge steps and vocab size 32000
Loaded 2,456 Hindi sentences
Loaded 1,234 Marathi sentences
Total corpus size: 3,690 sentences

BPE Training:
Found 15,432 unique words
Initial vocabulary size: 847
Training complete. Final vocabulary size: 32,000
Total merges performed: 31,153

WordPiece Training:
Found 15,432 unique words  
Initial vocabulary size: 848
Training complete. Final vocabulary size: 32,000
```

### Example Tokenization Results
```
Test 1: प्राकृतिक भाषा प्रसंस्करण
BPE tokens (8): ['प्रा', 'कृतिक', 'भा', 'षा', 'प्र', 'संस्', 'क', 'रण']
WordPiece tokens (6): ['प्राकृतिक', 'भाषा', 'प्र', '##संस्', '##क', '##रण']

Test 2: मशीन लर्निंग एल्गोरिदम
BPE tokens (9): ['म', 'शीन', 'ल', 'र्निंग', 'एल्', 'गो', 'रि', 'द', 'म']
WordPiece tokens (7): ['मशीन', 'लर्निंग', 'एल्', '##गो', '##रि', '##द', '##म']
```

## Performance Characteristics

### Computational Complexity
- **BPE Training**: O(N × M) where N = corpus size, M = merge operations
- **WordPiece Training**: O(V × P) where V = vocabulary, P = candidate pieces
- **Encoding**: O(W × L) where W = words, L = average word length

### Memory Usage
- **BPE**: O(V + M) for vocabulary and merge rules
- **WordPiece**: O(V + C) for vocabulary and candidate scores
- **Efficient Storage**: Optimized data structures for large vocabularies

### Quality Metrics
- **Compression Ratio**: Original words / subword tokens
- **Vocabulary Efficiency**: Coverage with minimal vocabulary
- **OOV Handling**: Ability to tokenize unseen words

## Key Insights

### BPE Advantages
- **Deterministic**: Consistent tokenization based on merge order
- **Simple**: Easy to understand and implement
- **Efficient**: Fast training and encoding
- **Language Agnostic**: Works across different languages

### WordPiece Advantages  
- **Probabilistic**: Uses statistical evidence for vocabulary building
- **Contextual**: Continuation markers preserve word structure
- **Quality**: Often produces more linguistically meaningful pieces
- **Robust**: Better handling of rare and compound words

### Multilingual Considerations
- **Script Support**: Comprehensive Unicode range handling
- **Morphology**: Adapts to different morphological complexity
- **Shared Vocabulary**: Enables cross-lingual representations
- **Cultural Sensitivity**: Respects language-specific patterns

## Applications
- **Neural Machine Translation**: Subword-level translation models
- **Language Modeling**: BERT, GPT, and transformer architectures  
- **Speech Recognition**: Handling out-of-vocabulary words
- **Information Retrieval**: Robust text matching and search
- **Cross-lingual NLP**: Multilingual model development

## Requirements
- `collections` (defaultdict, Counter)
- `json` (data loading)
- `re` (pattern matching)
- `os` (file operations)

## Educational Value
- Demonstrates modern tokenization techniques used in state-of-the-art NLP models
- Shows practical handling of multilingual text processing challenges
- Illustrates algorithmic differences between frequency-based and likelihood-based approaches
- Provides foundation for understanding transformer-based language models
- Bridges traditional NLP preprocessing with modern deep learning pipelines