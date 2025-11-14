# POS Tagging with HMM and Viterbi Decoding - Lab 10 Results

## Implementation Overview

This implementation creates a Part-of-Speech (POS) tagger using Hidden Markov Models (HMM) with the following components:

### 1. Data Splitting (K-fold Cross Validation)
- **K = 5 folds** (configurable, K ≥ 3)
- Random shuffle with fixed seed for reproducibility
- Each fold uses 80% data for training, 20% for testing

### 2. HMM Model Components

#### a) Emission Probabilities
- P(word|tag) = probability of observing a word given a POS tag
- **Smoothing**: Add-one (Laplace) smoothing applied
- **Unknown Words**: Handled with special `<UNK>` token

#### b) Transition Probabilities
- P(tag_i|tag_{i-1}) = probability of transitioning from one tag to another
- **Special Tags**: `<START>` and `<END>` tokens for sentence boundaries
- **Smoothing**: Add-one smoothing for unseen tag transitions

### 3. Viterbi Decoding Algorithm
- **Dynamic Programming**: Finds most likely tag sequence
- **Log Probabilities**: Used to prevent numerical underflow
- **Backpointer**: Tracks optimal path for reconstruction
- **Time Complexity**: O(n × |T|²) where n = sentence length, |T| = number of tags

## Performance Results

### Cross-Validation Performance (5-fold)
```
Mean Accuracy: 85.35% ± 0.52%

Fold-wise Results:
Fold 1: 85.49%
Fold 2: 84.74%
Fold 3: 86.07%
Fold 4: 84.77%
Fold 5: 85.67%
```

### Overall Metrics
- **Accuracy**: 85.35%
- **Macro-averaged Precision**: 72.82%
- **Macro-averaged Recall**: 60.96%
- **Macro-averaged F1-score**: 63.66%
- **Micro-averaged F1-score**: 85.35%

### Top Performing POS Tags
| Tag | Precision | Recall | F1-score | Support |
|-----|-----------|--------|----------|---------|
| NN  | 80.74%    | 87.00% | 83.75%   | 13,166  |
| IN  | 85.93%    | 97.87% | 91.51%   | 9,857   |
| NNP | 81.02%    | 78.99% | 79.99%   | 9,410   |
| DT  | 79.68%    | 98.81% | 88.22%   | 8,165   |
| ,   | 94.23%    | 99.98% | 97.02%   | 4,886   |

## Key Features

### 1. Robust Data Parsing
- Handles malformed tokens gracefully
- Case-insensitive processing
- Proper handling of punctuation

### 2. Smoothing Techniques
- **Add-one smoothing** for both emission and transition probabilities
- **Unknown word handling** with `<UNK>` token
- Prevents zero probabilities that would break Viterbi

### 3. Comprehensive Evaluation
- **Per-tag metrics**: Precision, Recall, F1-score for each POS tag
- **Macro/Micro averaging**: Different perspectives on performance
- **Cross-validation**: Robust estimation with confidence intervals

### 4. Sample Predictions
```
Sentence: "The quick brown fox jumps over the lazy dog ."
Predicted: DT JJ NNS VBP VBN IN DT JJ NN .

Sentence: "I love natural language processing ."
Predicted: PRP VBP JJ NN NN .

Sentence: "Stock prices rose yesterday ."
Predicted: NN NNS VBD NN .
```

## Technical Details

### Model Architecture
1. **Training Phase**:
   - Parse tagged sentences into (word, tag) pairs
   - Count transition and emission frequencies
   - Apply smoothing and calculate probabilities

2. **Inference Phase**:
   - Use Viterbi algorithm to find optimal tag sequence
   - Handle out-of-vocabulary words
   - Return most likely tag sequence

### Performance Analysis
- **Strengths**: High accuracy on common tags (nouns, prepositions, determiners)
- **Challenges**: Lower performance on verbs and adjectives due to ambiguity
- **Consistency**: Low variance across folds indicates stable performance

### Computational Complexity
- **Training**: O(N × L) where N = number of sentences, L = average sentence length
- **Inference**: O(L × |T|²) per sentence where |T| = number of unique tags
- **Memory**: O(|V| × |T| + |T|²) for emission and transition tables

## Dataset Statistics
- **Total Sentences**: 3,914
- **Total Tokens**: ~94,000
- **Unique POS Tags**: 45
- **Vocabulary Size**: ~12,000 unique words

This implementation demonstrates a solid foundation for POS tagging with good performance on the Wall Street Journal corpus, achieving competitive accuracy while maintaining interpretability through the HMM framework.