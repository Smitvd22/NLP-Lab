# Lab 8: Naive Bayes Text Classification

## Overview
This lab implements a comprehensive Naive Bayes classifier for text classification, incorporating advanced feature extraction techniques and multiple probability models for robust text categorization.

## Files
- `naive_bayes.py` - Main Naive Bayes implementation with feature engineering
- `naive_bayes_detailed_analysis.py` - Extended analysis and evaluation

## Problem Domain: Message Classification

### Classes
- **Inform**: Informational messages (notifications, updates, facts)
- **Promo**: Promotional messages (offers, deals, advertisements) 
- **Reminder**: Reminder messages (appointments, deadlines, tasks)

### Sample Dataset
```python
raw_data = [
    ("Check out https://example.com for more info!", "Inform"),
    ("Order 3 items, get 1 free! Limited offer!!!", "Promo"),
    ("Your package #12345 will arrive tomorrow.", "Inform"),
    ("Win $1000 now, visit http://winbig.com!!!", "Promo"),
    ("Meeting at 3pm, don't forget to bring the files.", "Reminder"),
    # ... additional examples
]
```

## Feature Engineering

### 1. Regex-based Tokenizer
Enhanced tokenizer handling multiple text patterns:

```python
def tokenizer(text):
    # Patterns from Assignment 1
    url_pattern = r'https?://\S+|www\.\S+'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'
    number_pattern = r'\d+(?:\.\d+)?'
    word_pattern = r"\w+"
    punct_pattern = r'[^\w\s]'
```

### 2. Binary Features (Bernoulli Model)
```python
def extract_features(sentence):
    has_url = 1 if re.search(r'https?://|www\.', sentence) else 0
    has_number = 1 if re.search(r'\d', sentence) else 0
    has_exclaim = 1 if '!' in sentence else 0
    return {
        "has_url": has_url,
        "has_number": has_number, 
        "has_exclaim": has_exclaim,
        "bigrams": get_bigrams(tokens)
    }
```

### 3. Bigram Features (Multinomial Model)
```python
def get_bigrams(tokens):
    lower_tokens = [t.lower() for t in tokens]
    if len(lower_tokens) < 2: return []
    return [tuple(lower_tokens[i:i+2]) for i in range(len(lower_tokens)-1)]
```

## Naive Bayes Implementation

### Class Structure
```python
class NaiveBayesClassifier:
    def __init__(self, k=0.3):  # Laplace smoothing parameter
        self.k = k
        self.classes = set()
        self.class_counts = defaultdict(int)
        self.feature_counts = {
            "has_url": defaultdict(int),
            "has_number": defaultdict(int), 
            "has_exclaim": defaultdict(int),
            "bigrams": defaultdict(lambda: defaultdict(int))
        }
        self.total_docs_in_class = defaultdict(int)
        self.total_bigrams_in_class = defaultdict(int)
        self.bigram_vocab = set()
```

### Mathematical Foundations

#### 1. Prior Probability
```
P(Class) = Count(Documents_in_Class) / Total_Documents
```

#### 2. Binary Feature Probability (Bernoulli Model)
```
P(Feature=1 | Class) = (Count(Docs_with_Feature, Class) + k) / (Count(Docs_in_Class) + k×2)
```

#### 3. Bigram Probability (Multinomial Model) 
```
P(Bigram | Class) = (Count(Bigram, Class) + k) / (Total_Bigrams_in_Class + k×V)
```
where V = bigram vocabulary size

#### 4. Classification Formula
```
Class* = argmax_c [P(c) × ∏P(f_i | c) × ∏P(bigram_j | c)]
```

### Key Methods

#### Training Phase
```python
def train(self, data):
    for sentence, label in data:
        self.classes.add(label)
        self.class_counts[label] += 1
        
        feats = extract_features(sentence)
        
        # Binary feature counting
        if feats["has_url"]: self.feature_counts["has_url"][label] += 1
        if feats["has_number"]: self.feature_counts["has_number"][label] += 1
        if feats["has_exclaim"]: self.feature_counts["has_exclaim"][label] += 1
        
        # Bigram counting
        for bg in feats["bigrams"]:
            self.feature_counts["bigrams"][bg][label] += 1
            self.bigram_vocab.add(bg)
```

#### Prediction Phase
```python
def predict(self, sentence):
    feats = extract_features(sentence)
    scores = {}
    
    for c in self.classes:
        # Start with log prior
        log_prob = math.log(self.priors[c])
        
        # Add binary feature probabilities
        for fname in ["has_url", "has_number", "has_exclaim"]:
            val = feats[fname]
            p = self.get_feature_prob(fname, c, val)
            log_prob += math.log(p)
        
        # Add bigram probabilities  
        for bg in feats["bigrams"]:
            p = self.get_bigram_prob(bg, c)
            log_prob += math.log(p)
        
        scores[c] = log_prob
    
    return max(scores, key=scores.get)
```

## Smoothing Techniques

### Laplace Smoothing (Add-k)
Uses k=0.3 for balance between smoothing strength and data fidelity:

#### Binary Features
```python
def get_feature_prob(self, feature_name, class_label, val=1):
    count_f = self.feature_counts[feature_name][class_label]
    total_c = self.total_docs_in_class[class_label]
    prob_present = (count_f + self.k) / (total_c + (self.k * 2))
    return prob_present if val == 1 else (1 - prob_present)
```

#### Bigrams  
```python
def get_bigram_prob(self, bigram, class_label):
    count_w = self.feature_counts["bigrams"][bigram][class_label]
    total_w = self.total_bigrams_in_class[class_label]
    vocab_size = len(self.bigram_vocab)
    return (count_w + self.k) / (total_w + (self.k * vocab_size))
```

## Detailed Analysis Features

### 1. Probability Decomposition
The classifier shows step-by-step probability calculations:

```
--- Predicting for: 'You will get an exclusive offer in the meeting!' ---

Class: Inform (Log Score: -15.2341)
Class: Promo (Log Score: -12.8976)  
Class: Reminder (Log Score: -16.7823)

>>> FINAL PREDICTION: Promo <<<
```

### 2. Feature Analysis
```python
print(f"P(has_url=1 | Inform)    = 0.1250")
print(f"P(has_number=1 | Promo)  = 0.7500") 
print(f"P(has_exclaim=1 | Promo) = 0.8750")
```

### 3. Training Statistics
- Documents per class distribution
- Feature frequency analysis  
- Bigram vocabulary statistics
- Cross-validation performance

## Advanced Features

### 1. Multi-Model Architecture
- **Bernoulli Model**: For binary presence/absence features
- **Multinomial Model**: For count-based bigram features
- **Hybrid Approach**: Combines both for robust classification

### 2. Robust Text Processing
- **URL Detection**: Handles various URL formats
- **Number Recognition**: Decimal and integer patterns
- **Punctuation Analysis**: Exclamation mark patterns
- **Case Normalization**: Consistent bigram processing

### 3. Evaluation Capabilities
- **Prediction Confidence**: Log probability scores
- **Feature Importance**: Individual feature contributions
- **Error Analysis**: Misclassification investigation
- **Performance Metrics**: Accuracy, precision, recall

## Usage

### Basic Classification
```python
# Initialize and train
nb = NaiveBayesClassifier(k=0.3)
nb.train(raw_data)

# Predict new message
test_sentence = "You will get an exclusive offer in the meeting!"
prediction = nb.predict(test_sentence)
print(f"Prediction: {prediction}")
```

### Running Complete Analysis
```bash
python naive_bayes.py
```

**Process:**
1. Display preprocessed training data with tokenization
2. Train Naive Bayes classifier on labeled dataset
3. Show feature probability examples for each class
4. Demonstrate prediction process with detailed scoring
5. Output final classification result

## Key Implementation Insights

### 1. Independence Assumption
Despite feature dependencies in real text, Naive Bayes assumes:
```
P(features | class) = ∏ P(feature_i | class)
```

### 2. Log-Space Computation
Uses logarithms to prevent numerical underflow:
```
log P(class | features) ∝ log P(class) + Σ log P(feature_i | class)
```

### 3. Smoothing Necessity
Essential for handling:
- Unseen bigrams in test data
- Zero probabilities that would eliminate candidates
- Balanced probability distribution

### 4. Feature Engineering Impact
- **Binary features** capture document-level properties
- **Bigrams** capture local context and word order
- **Combination** provides comprehensive text representation

## Applications

### 1. Email Classification
- Spam detection
- Category assignment
- Priority classification

### 2. Social Media Analysis
- Sentiment analysis preprocessing
- Content categorization
- User intent detection

### 3. Customer Support
- Ticket routing
- Query classification
- Response prioritization

### 4. Content Management
- Document categorization
- News classification
- Product description analysis

## Performance Characteristics

### Computational Complexity
- **Training**: O(N × F) where N = documents, F = features per document
- **Prediction**: O(F × C) where C = number of classes
- **Memory**: O(V × C) where V = vocabulary size

### Advantages
- Fast training and prediction
- Handles missing features naturally
- Works well with small datasets
- Interpretable probability outputs

### Limitations
- Strong independence assumption
- Sensitive to correlated features
- Requires smoothing for robustness

## Requirements
- `collections` (defaultdict, Counter)
- `math` (logarithmic computations)
- `re` (regular expressions)

## Educational Value
- Demonstrates fundamental classification concepts
- Shows practical text preprocessing techniques
- Illustrates probability estimation methods
- Provides foundation for advanced NLP classifiers
- Combines theoretical concepts with hands-on implementation