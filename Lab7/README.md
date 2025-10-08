# Lab 7: PMI, TF-IDF, and Nearest Neighbor Analysis

## Overview
This lab implements three main components:
1. **PMI (Pointwise Mutual Information)** computation for bigrams
2. **TF-IDF Vectorization** of sentences
3. **Nearest Neighbor Search** using TF-IDF vectors

## Data
- **Source**: Tokenized Hindi and Marathi sentences from Lab1
- **Language Models**: Unigram and bigram models from Lab4
- **Total Sentences**: 105 (66 Hindi + 39 Marathi)
- **Split**: 84 training, 10 validation, 11 test sentences

## Implementation Results

### 1. PMI Analysis (`pmi.py`)
**Purpose**: Compute PMI scores for all bigrams in validation and test sets using trained language models.

**Key Results**:
- Hindi model: 1,300 unigrams, 1,299 bigrams
- Marathi model: 427 unigrams, 426 bigrams
- Generated PMI scores for both validation and test sets using both language models

**Top PMI Bigrams (Hindi model)**:
- Validation: "रहने वाले", "कुमार विश्वकर्मा", "गुलाबी मीनाकारी" (PMI ≈ 7.17)
- Test: "सिर्फ बेसिक्स", "शांत चित्त", "चित्त बने" (PMI ≈ 7.17)

**Output Files**:
- `pmi_validation_scores.csv` - PMI scores for validation set
- `pmi_test_scores.csv` - PMI scores for test set
- `pmi_validation_marathi_scores.csv` - PMI scores using Marathi model
- `pmi_test_marathi_scores.csv` - PMI scores using Marathi model

### 2. TF-IDF Vectorization (`tf-idf.py`)
**Purpose**: Convert all sentences to TF-IDF vectors using IDF weights learned from training data.

**Key Results**:
- Vocabulary size: 797 unique terms (from training data)
- Matrix dimensions: Train(84×797), Val(10×797), Test(11×797)
- Sparsity: Training(1.86%), Validation(1.18%), Test(1.02%)

**Features**:
- Term Frequency (TF) = count / document_length
- Inverse Document Frequency (IDF) = log((N+1)/(df+1)) + 1
- TF-IDF = TF × IDF
- Efficient sparse matrix representation

**Output Files**:
- `tfidf_train.npz` - Training TF-IDF matrix
- `tfidf_val.npz` - Validation TF-IDF matrix  
- `tfidf_test.npz` - Test TF-IDF matrix
- `vocab.json` - Vocabulary mapping
- `idf.npy` - IDF weights
- `tfidf_examples.txt` - Sample vectorization examples

### 3. Nearest Neighbor Search (`nearest_neighbour.py`)
**Purpose**: Find nearest neighbors for each validation/test sentence using cosine similarity.

**Key Results**:
- Vocabulary alignment: 108 common features (after filtering)
- Similarity metric: Cosine similarity
- Search scope: Top-5 nearest neighbors from training set

**Features**:
- Handles vocabulary mismatch between train/test sets
- Efficient batch processing for large datasets
- Cosine similarity: sim(A,B) = A·B / (||A|| × ||B||)
- Statistical analysis of similarity distributions

**Output Files**:
- `val_neighbors.txt` - Nearest neighbors for validation set
- `test_neighbors.txt` - Nearest neighbors for test set

## Key Statistics

### Similarity Analysis
- Mean similarity: 0.074
- Standard deviation: 0.133
- Max similarity: 1.000 (perfect match)
- Min similarity: 0.000 (no common terms)

### Example Results
**Query**: "मछोदरी के रहने वाले रमेश कुमार विश्वकर्मा को चांदी की गुलाबी मीनाकारी..."
**Best Match** (Score: 0.728): "रमेश के साथ ही देशभर के 40 हस्तशिल्पियों को राष्ट्रीय पुरस्कार..."

## Technical Approach

### PMI Computation
```python
PMI(w1, w2) = log(P(w1,w2) / (P(w1) × P(w2)))
```
Where:
- P(w1,w2) = bigram_count / total_bigrams
- P(w1) = unigram_count / total_unigrams

### TF-IDF Calculation
```python
TF(t,d) = count(t,d) / |d|
IDF(t) = log((N+1) / (df(t)+1)) + 1
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### Cosine Similarity
```python
cosine_sim(A,B) = A·B / (||A||₂ × ||B||₂)
```

## Files Generated
- **Train/Val/Test splits**: `train.txt`, `val.txt`, `test.txt`
- **PMI results**: 4 CSV files with PMI scores
- **TF-IDF data**: 3 compressed matrices + vocabulary + IDF weights
- **Neighbor results**: 2 text files with nearest neighbor rankings
- **Examples**: Sample TF-IDF vectors for inspection

## Usage
```bash
# Run in sequence:
python pmi.py        # Creates splits and computes PMI
python tf-idf.py     # Vectorizes all sentences
python nearest_neighbour.py  # Finds nearest neighbors
```

All scripts are designed to work with the existing Lab1 (tokenized data) and Lab4 (language models) structure.