"""
Lab 7 - TF-IDF Vectorization
Vectorize all sentences in training, validation, and testing data using TF-IDF.
For validation and testing data, use IDF scores learned from training data.
"""

import json
import math
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict, Counter

# Paths
LAB7_PATH = Path(__file__).parent
TRAIN_PATH = LAB7_PATH / "train.txt"
VAL_PATH = LAB7_PATH / "val.txt"
TEST_PATH = LAB7_PATH / "test.txt"

# Output files
TFIDF_TRAIN_NPZ = LAB7_PATH / "tfidf_train.npz"
TFIDF_VAL_NPZ = LAB7_PATH / "tfidf_val.npz"
TFIDF_TEST_NPZ = LAB7_PATH / "tfidf_test.npz"
VOCAB_PATH = LAB7_PATH / "vocab.json"
IDF_PATH = LAB7_PATH / "idf.npy"
TFIDF_EXAMPLES_PATH = LAB7_PATH / "tfidf_examples.txt"

def load_sentences(filepath):
    """Load sentences from a text file."""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences

def build_vocabulary(sentences):
    """Build vocabulary from sentences."""
    vocab = {}
    vocab_index = 0
    
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = vocab_index
                vocab_index += 1
    
    return vocab

def compute_tf_matrix(sentences, vocab):
    """Compute Term Frequency (TF) matrix."""
    n_docs = len(sentences)
    n_vocab = len(vocab)
    
    # Use lists to build sparse matrix efficiently
    rows = []
    cols = []
    data = []
    
    for doc_id, sentence in enumerate(sentences):
        tokens = sentence.split()
        if not tokens:
            continue
            
        # Count term frequencies in this document
        term_counts = Counter(tokens)
        doc_length = len(tokens)
        
        # Add TF values to sparse matrix
        for term, count in term_counts.items():
            if term in vocab:
                term_id = vocab[term]
                tf = count / doc_length  # TF = count / document_length
                
                rows.append(doc_id)
                cols.append(term_id)
                data.append(tf)
    
    # Create sparse matrix
    tf_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_docs, n_vocab), dtype=np.float32)
    return tf_matrix

def compute_idf(tf_matrix):
    """Compute Inverse Document Frequency (IDF) from TF matrix."""
    # Convert to CSR for efficient column operations
    tf_csr = tf_matrix.tocsr()
    
    # Count documents containing each term (document frequency)
    df = np.asarray((tf_csr > 0).sum(axis=0)).ravel()
    
    # Compute IDF: log(N / df) + 1 (adding 1 for smoothing)
    n_docs = tf_matrix.shape[0]
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    
    return idf.astype(np.float32)

def apply_idf_to_tf(tf_matrix, idf):
    """Apply IDF weights to TF matrix to get TF-IDF matrix."""
    tf_coo = tf_matrix.tocoo()
    
    # Multiply TF values by corresponding IDF values
    tfidf_data = tf_coo.data * idf[tf_coo.col]
    
    # Create TF-IDF matrix
    tfidf_matrix = sp.coo_matrix(
        (tfidf_data, (tf_coo.row, tf_coo.col)), 
        shape=tf_coo.shape, 
        dtype=np.float32
    )
    
    return tfidf_matrix.tocsr()

def save_sparse_matrix(matrix, filepath):
    """Save sparse matrix to compressed numpy format."""
    matrix_csr = matrix.tocsr()
    np.savez_compressed(
        filepath,
        data=matrix_csr.data,
        indices=matrix_csr.indices,
        indptr=matrix_csr.indptr,
        shape=np.array(matrix_csr.shape, dtype=np.int64)
    )

def save_vocabulary(vocab, filepath):
    """Save vocabulary to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def save_tfidf_examples(vocab, tfidf_matrix, sentences, filepath, n_examples=5):
    """Save TF-IDF examples to text file for inspection."""
    # Create inverse vocabulary
    inv_vocab = {idx: token for token, idx in vocab.items()}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("TF-IDF Vectorization Examples\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Vocabulary size: {len(vocab)}\n")
        f.write(f"Number of documents: {tfidf_matrix.shape[0]}\n")
        f.write(f"TF-IDF matrix shape: {tfidf_matrix.shape}\n\n")
        
        f.write("Sample vocabulary (first 50 terms):\n")
        sample_terms = list(vocab.keys())[:50]
        f.write(", ".join(sample_terms) + "\n\n")
        
        f.write(f"TF-IDF vectors for first {n_examples} documents:\n")
        f.write("-" * 50 + "\n")
        
        for i in range(min(n_examples, len(sentences))):
            f.write(f"\nDocument {i + 1}:\n")
            f.write(f"Sentence: {sentences[i]}\n")
            
            # Get TF-IDF vector for this document
            doc_vector = tfidf_matrix.getrow(i)
            
            if doc_vector.nnz == 0:
                f.write("TF-IDF: (empty vector)\n")
                continue
            
            # Get non-zero terms and their TF-IDF scores
            doc_coo = doc_vector.tocoo()
            term_scores = []
            
            for j in range(doc_coo.nnz):
                term_id = doc_coo.col[j]
                score = doc_coo.data[j]
                term = inv_vocab.get(term_id, f"UNK_{term_id}")
                term_scores.append((term, score))
            
            # Sort by TF-IDF score (descending)
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write("TF-IDF (top terms): ")
            top_terms = term_scores[:10]  # Show top 10 terms
            terms_str = ", ".join([f"{term}:{score:.4f}" for term, score in top_terms])
            f.write(terms_str + "\n")

def main():
    print("Loading sentences from train, validation, and test sets...")
    
    # Load sentences
    train_sentences = load_sentences(TRAIN_PATH)
    val_sentences = load_sentences(VAL_PATH)
    test_sentences = load_sentences(TEST_PATH)
    
    print(f"Loaded sentences - Train: {len(train_sentences)}, Val: {len(val_sentences)}, Test: {len(test_sentences)}")
    
    # Build vocabulary from training data only
    print("Building vocabulary from training data...")
    vocab = build_vocabulary(train_sentences)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Compute TF matrices
    print("Computing TF matrices...")
    train_tf = compute_tf_matrix(train_sentences, vocab)
    val_tf = compute_tf_matrix(val_sentences, vocab)
    test_tf = compute_tf_matrix(test_sentences, vocab)
    
    print(f"TF matrix shapes - Train: {train_tf.shape}, Val: {val_tf.shape}, Test: {test_tf.shape}")
    
    # Compute IDF from training data
    print("Computing IDF from training data...")
    idf = compute_idf(train_tf)
    print(f"IDF vector shape: {idf.shape}")
    
    # Apply IDF to get TF-IDF matrices
    print("Computing TF-IDF matrices...")
    train_tfidf = apply_idf_to_tf(train_tf, idf)
    val_tfidf = apply_idf_to_tf(val_tf, idf)
    test_tfidf = apply_idf_to_tf(test_tf, idf)
    
    print(f"TF-IDF matrix shapes - Train: {train_tfidf.shape}, Val: {val_tfidf.shape}, Test: {test_tfidf.shape}")
    
    # Save TF-IDF matrices
    print("Saving TF-IDF matrices...")
    save_sparse_matrix(train_tfidf, TFIDF_TRAIN_NPZ)
    save_sparse_matrix(val_tfidf, TFIDF_VAL_NPZ)
    save_sparse_matrix(test_tfidf, TFIDF_TEST_NPZ)
    
    # Save vocabulary and IDF
    print("Saving vocabulary and IDF...")
    save_vocabulary(vocab, VOCAB_PATH)
    np.save(IDF_PATH, idf)
    
    # Save examples for inspection
    print("Saving TF-IDF examples...")
    save_tfidf_examples(vocab, train_tfidf, train_sentences, TFIDF_EXAMPLES_PATH)
    
    print("\nTF-IDF vectorization completed!")
    print(f"Files saved:")
    print(f"  - TF-IDF matrices: {TFIDF_TRAIN_NPZ}, {TFIDF_VAL_NPZ}, {TFIDF_TEST_NPZ}")
    print(f"  - Vocabulary: {VOCAB_PATH}")
    print(f"  - IDF weights: {IDF_PATH}")
    print(f"  - Examples: {TFIDF_EXAMPLES_PATH}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Training TF-IDF sparsity: {train_tfidf.nnz / (train_tfidf.shape[0] * train_tfidf.shape[1]):.6f}")
    print(f"  - Validation TF-IDF sparsity: {val_tfidf.nnz / (val_tfidf.shape[0] * val_tfidf.shape[1]):.6f}")
    print(f"  - Test TF-IDF sparsity: {test_tfidf.nnz / (test_tfidf.shape[0] * test_tfidf.shape[1]):.6f}")

if __name__ == "__main__":
    main()
