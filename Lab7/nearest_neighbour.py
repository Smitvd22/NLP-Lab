"""
Lab 7 - Nearest Neighbor Search using TF-IDF
For each sentence in validation and testing sets, find its nearest neighbor
in the training set using TF-IDF vectors and cosine similarity.
"""

import numpy as np
import scipy.sparse as sp
import json
from pathlib import Path

# Paths
LAB7_PATH = Path(__file__).parent
TRAIN_PATH = LAB7_PATH / "train.txt"
VAL_PATH = LAB7_PATH / "val.txt"
TEST_PATH = LAB7_PATH / "test.txt"

# TF-IDF files
TFIDF_TRAIN_NPZ = LAB7_PATH / "tfidf_train.npz"
TFIDF_VAL_NPZ = LAB7_PATH / "tfidf_val.npz"
TFIDF_TEST_NPZ = LAB7_PATH / "tfidf_test.npz"
VOCAB_PATH = LAB7_PATH / "vocab.json"

# Output files
VAL_NEIGHBORS_PATH = LAB7_PATH / "val_neighbors.txt"
TEST_NEIGHBORS_PATH = LAB7_PATH / "test_neighbors.txt"

def load_sentences(filepath):
    """Load sentences from text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_sparse_matrix(filepath):
    """Load sparse matrix from npz file."""
    npz = np.load(filepath, allow_pickle=False)
    data = npz['data']
    indices = npz['indices'].astype(np.int32)
    indptr = npz['indptr'].astype(np.int32)
    shape = tuple(npz['shape'].astype(int).tolist())
    
    return sp.csr_matrix((data, indices, indptr), shape=shape)

def load_vocabulary(filepath):
    """Load vocabulary from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_row_norms(matrix):
    """Compute L2 norms for each row of the sparse matrix."""
    # Element-wise square
    squared = matrix.multiply(matrix)
    # Sum along axis 1 (columns)
    row_sums = np.asarray(squared.sum(axis=1)).ravel()
    # Square root to get norms
    norms = np.sqrt(row_sums + 1e-12)  # Add small epsilon for numerical stability
    return norms

def find_nearest_neighbors(train_matrix, train_norms, query_matrix, query_norms, 
                          train_sentences, query_sentences, output_path, 
                          top_k=5, batch_size=100):
    """
    Find nearest neighbors using cosine similarity.
    
    Args:
        train_matrix: Training TF-IDF matrix (n_train x vocab_size)
        train_norms: L2 norms of training vectors
        query_matrix: Query TF-IDF matrix (n_query x vocab_size)
        query_norms: L2 norms of query vectors
        train_sentences: List of training sentences
        query_sentences: List of query sentences
        output_path: Path to save results
        top_k: Number of nearest neighbors to find
        batch_size: Batch size for processing queries
    """
    n_queries = query_matrix.shape[0]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Nearest Neighbor Search Results\n")
        f.write(f"Query set size: {n_queries}\n")
        f.write(f"Training set size: {train_matrix.shape[0]}\n")
        f.write(f"Top-K neighbors: {top_k}\n")
        f.write("="*80 + "\n\n")
        
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_queries = query_matrix[start_idx:end_idx]
            batch_norms = query_norms[start_idx:end_idx]
            
            # Compute cosine similarities: batch_queries @ train_matrix.T
            similarities = batch_queries.dot(train_matrix.T).toarray()  # (batch_size, n_train)
            
            # Normalize by row norms to get cosine similarity
            # similarities[i, j] = dot(query_i, train_j) / (norm(query_i) * norm(train_j))
            denominators = np.outer(batch_norms, train_norms)
            similarities = similarities / (denominators + 1e-12)
            
            # Find top-k nearest neighbors for each query in the batch
            k = min(top_k, similarities.shape[1])
            
            # Use argpartition for efficiency (partial sorting)
            top_indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
            
            # Process each query in the batch
            for i in range(similarities.shape[0]):
                query_idx = start_idx + i
                candidates = top_indices[i]
                scores = similarities[i, candidates]
                
                # Sort candidates by similarity score (descending)
                sorted_order = np.argsort(-scores)
                sorted_indices = candidates[sorted_order]
                sorted_scores = scores[sorted_order]
                
                # Write results
                f.write(f"Query [{query_idx}]: {query_sentences[query_idx]}\n")
                
                for rank, (neighbor_idx, score) in enumerate(zip(sorted_indices, sorted_scores), 1):
                    neighbor_idx = int(neighbor_idx)
                    score = float(score)
                    neighbor_sentence = train_sentences[neighbor_idx]
                    
                    f.write(f"  Neighbor {rank}: ")
                    f.write(f"TrainIndex={neighbor_idx:5d} ")
                    f.write(f"Score={score:.6f} ")
                    f.write(f"Sentence={neighbor_sentence}\n")
                
                f.write("\n")
            
            # Progress update
            print(f"Processed queries {start_idx + 1}-{end_idx} / {n_queries}")

def analyze_similarity_distribution(train_matrix, query_matrix, train_norms, query_norms, sample_size=100):
    """Analyze the distribution of similarity scores for insight."""
    print("Analyzing similarity score distribution...")
    
    # Sample a subset for analysis
    n_queries = min(sample_size, query_matrix.shape[0])
    sample_indices = np.random.choice(query_matrix.shape[0], n_queries, replace=False)
    
    sample_queries = query_matrix[sample_indices]
    sample_norms = query_norms[sample_indices]
    
    # Compute similarities
    similarities = sample_queries.dot(train_matrix.T).toarray()
    denominators = np.outer(sample_norms, train_norms)
    similarities = similarities / (denominators + 1e-12)
    
    # Statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    
    print(f"Similarity statistics (sample of {n_queries} queries):")
    print(f"  Mean: {mean_sim:.6f}")
    print(f"  Std:  {std_sim:.6f}")
    print(f"  Max:  {max_sim:.6f}")
    print(f"  Min:  {min_sim:.6f}")
    
    return similarities

def main():
    print("Loading sentences...")
    train_sentences = load_sentences(TRAIN_PATH)
    val_sentences = load_sentences(VAL_PATH)
    test_sentences = load_sentences(TEST_PATH)
    
    print(f"Loaded sentences:")
    print(f"  Train: {len(train_sentences)}")
    print(f"  Validation: {len(val_sentences)}")
    print(f"  Test: {len(test_sentences)}")
    
    print("\nLoading TF-IDF matrices...")
    train_tfidf = load_sparse_matrix(TFIDF_TRAIN_NPZ)
    val_tfidf = load_sparse_matrix(TFIDF_VAL_NPZ)
    test_tfidf = load_sparse_matrix(TFIDF_TEST_NPZ)
    
    print(f"TF-IDF matrix shapes:")
    print(f"  Train: {train_tfidf.shape}")
    print(f"  Validation: {val_tfidf.shape}")
    print(f"  Test: {test_tfidf.shape}")
    
    # Handle vocabulary mismatch by filtering columns
    # Only keep columns (features) that appear in both train and query sets
    print("\nHandling vocabulary alignment...")
    
    val_vocab_cols = set(val_tfidf.indices)
    test_vocab_cols = set(test_tfidf.indices)
    all_query_cols = val_vocab_cols.union(test_vocab_cols)
    
    # Keep intersection of training and query vocabularies
    train_vocab_cols = set(train_tfidf.indices)
    common_cols = train_vocab_cols.intersection(all_query_cols)
    
    if len(common_cols) < len(train_vocab_cols):
        print(f"Vocabulary alignment: {len(common_cols)} common features out of {len(train_vocab_cols)} training features")
    
    # Filter matrices to common vocabulary
    def filter_matrix_columns(matrix, valid_cols):
        """Filter sparse matrix to keep only specified columns."""
        if len(valid_cols) == matrix.shape[1]:
            return matrix  # No filtering needed
        
        coo = matrix.tocoo()
        mask = np.isin(coo.col, list(valid_cols))
        
        if not np.any(mask):
            # No common columns - return empty matrix
            return sp.csr_matrix((matrix.shape[0], len(valid_cols)))
        
        filtered_data = coo.data[mask]
        filtered_row = coo.row[mask]
        filtered_col = coo.col[mask]
        
        # Remap column indices
        col_mapping = {col: idx for idx, col in enumerate(sorted(valid_cols))}
        new_col = np.array([col_mapping[col] for col in filtered_col])
        
        return sp.coo_matrix(
            (filtered_data, (filtered_row, new_col)), 
            shape=(matrix.shape[0], len(valid_cols))
        ).tocsr()
    
    common_cols_list = sorted(list(common_cols))
    train_tfidf_filtered = filter_matrix_columns(train_tfidf, common_cols_list)
    val_tfidf_filtered = filter_matrix_columns(val_tfidf, common_cols_list)
    test_tfidf_filtered = filter_matrix_columns(test_tfidf, common_cols_list)
    
    print(f"Filtered matrix shapes:")
    print(f"  Train: {train_tfidf_filtered.shape}")
    print(f"  Validation: {val_tfidf_filtered.shape}")
    print(f"  Test: {test_tfidf_filtered.shape}")
    
    # Compute row norms for cosine similarity
    print("\nComputing row norms...")
    train_norms = compute_row_norms(train_tfidf_filtered)
    val_norms = compute_row_norms(val_tfidf_filtered)
    test_norms = compute_row_norms(test_tfidf_filtered)
    
    print(f"Row norms computed:")
    print(f"  Train norms - mean: {np.mean(train_norms):.6f}, std: {np.std(train_norms):.6f}")
    print(f"  Val norms - mean: {np.mean(val_norms):.6f}, std: {np.std(val_norms):.6f}")
    print(f"  Test norms - mean: {np.mean(test_norms):.6f}, std: {np.std(test_norms):.6f}")
    
    # Analyze similarity distribution (optional)
    if len(val_sentences) > 0:
        analyze_similarity_distribution(
            train_tfidf_filtered, val_tfidf_filtered, 
            train_norms, val_norms, sample_size=50
        )
    
    # Find nearest neighbors for validation set
    if len(val_sentences) > 0:
        print(f"\nFinding nearest neighbors for validation set...")
        find_nearest_neighbors(
            train_tfidf_filtered, train_norms,
            val_tfidf_filtered, val_norms,
            train_sentences, val_sentences,
            VAL_NEIGHBORS_PATH,
            top_k=5, batch_size=50
        )
        print(f"Validation neighbors saved to: {VAL_NEIGHBORS_PATH}")
    
    # Find nearest neighbors for test set
    if len(test_sentences) > 0:
        print(f"\nFinding nearest neighbors for test set...")
        find_nearest_neighbors(
            train_tfidf_filtered, train_norms,
            test_tfidf_filtered, test_norms,
            train_sentences, test_sentences,
            TEST_NEIGHBORS_PATH,
            top_k=5, batch_size=50
        )
        print(f"Test neighbors saved to: {TEST_NEIGHBORS_PATH}")
    
    print("\nNearest neighbor search completed!")
    print(f"Results saved to:")
    print(f"  - Validation: {VAL_NEIGHBORS_PATH}")
    print(f"  - Test: {TEST_NEIGHBORS_PATH}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
