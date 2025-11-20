"""
Lab 7 - Bonus Question: Extended Nearest Neighbor Search
Extended implementation that finds nearest neighbors for training data as well,
with optimizations for handling large datasets efficiently.

For each sentence in training, validation and testing sets, find its nearest neighbor
using TF-IDF vectors and cosine similarity with computational optimizations.
"""

import numpy as np
import scipy.sparse as sp
import json
from pathlib import Path
import time
from sklearn.metrics.pairwise import cosine_similarity
import psutil
import gc

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
TRAIN_NEIGHBORS_PATH = LAB7_PATH / "train_neighbors.txt"
VAL_NEIGHBORS_PATH = LAB7_PATH / "val_neighbors_bonus.txt"
TEST_NEIGHBORS_PATH = LAB7_PATH / "test_neighbors_bonus.txt"
OPERATIONS_ANALYSIS_PATH = LAB7_PATH / "operations_analysis.txt"

class MemoryMonitor:
    """Monitor memory usage during computation."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def log_memory(operation_name):
        """Log current memory usage."""
        memory = MemoryMonitor.get_memory_usage()
        print(f"Memory after {operation_name}: {memory:.1f} MB")

class EfficientNearestNeighbors:
    """Efficient nearest neighbor search with memory and computation optimizations."""
    
    def __init__(self, chunk_size=1000, top_k=5):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.operation_count = 0
        self.similarity_computations = 0
        
    def load_sentences(self, filepath):
        """Load sentences from text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_sparse_matrix(self, filepath):
        """Load sparse matrix from npz file."""
        npz = np.load(filepath, allow_pickle=False)
        data = npz['data']
        indices = npz['indices'].astype(np.int32)
        indptr = npz['indptr'].astype(np.int32)
        shape = tuple(npz['shape'].astype(int).tolist())
        
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    def normalize_matrix(self, matrix):
        """
        L2 normalize rows of sparse matrix for cosine similarity.
        This allows us to compute cosine similarity as simple dot product.
        """
        print("Normalizing matrix for cosine similarity...")
        
        # Compute row norms
        squared = matrix.multiply(matrix)
        row_sums = np.asarray(squared.sum(axis=1)).ravel()
        norms = np.sqrt(row_sums + 1e-12)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Create diagonal matrix of inverse norms
        norm_inv = sp.diags(1.0 / norms, format='csr')
        
        # Normalize: each row divided by its norm
        normalized = norm_inv @ matrix
        
        return normalized

    def chunked_similarity_search(self, query_matrix, train_matrix, 
                                query_sentences, train_sentences,
                                exclude_self=False, query_start_idx=0):
        """
        Perform similarity search in chunks to manage memory usage.
        
        Args:
            query_matrix: Query TF-IDF matrix (normalized)
            train_matrix: Training TF-IDF matrix (normalized)  
            query_sentences: List of query sentences
            train_sentences: List of training sentences
            exclude_self: Whether to exclude self-matches (for train-train search)
            query_start_idx: Starting index for queries (for train-train search)
        
        Returns:
            List of (query_idx, neighbors_list) tuples
        """
        n_queries = query_matrix.shape[0]
        n_train = train_matrix.shape[0]
        results = []
        
        print(f"Processing {n_queries} queries against {n_train} training samples...")
        
        for chunk_start in range(0, n_queries, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_queries)
            chunk_size = chunk_end - chunk_start
            
            # Get chunk of query vectors
            query_chunk = query_matrix[chunk_start:chunk_end]
            
            print(f"Processing chunk {chunk_start+1}-{chunk_end} / {n_queries}")
            
            # Method 1: Use sklearn's cosine_similarity (memory efficient for large datasets)
            if n_train > 10000:  # Use sklearn for very large training sets
                similarities = cosine_similarity(query_chunk, train_matrix)
                self.similarity_computations += chunk_size * n_train
                
            # Method 2: Manual dot product (faster for smaller datasets)
            else:
                # Since matrices are normalized, cosine similarity = dot product
                similarities = query_chunk.dot(train_matrix.T).toarray()
                self.similarity_computations += chunk_size * n_train
            
            # Process each query in the chunk
            for i in range(chunk_size):
                global_query_idx = chunk_start + i
                query_similarities = similarities[i]
                
                # Exclude self-match if needed
                if exclude_self:
                    actual_query_idx = query_start_idx + global_query_idx
                    if actual_query_idx < len(query_similarities):
                        query_similarities[actual_query_idx] = -1.0  # Set to minimum
                
                # Find top-k neighbors efficiently
                if self.top_k >= len(query_similarities):
                    # If k >= n, sort all
                    top_indices = np.argsort(-query_similarities)
                    top_scores = query_similarities[top_indices]
                else:
                    # Use argpartition for efficiency (O(n) vs O(n log n))
                    top_indices = np.argpartition(-query_similarities, self.top_k-1)[:self.top_k]
                    top_scores = query_similarities[top_indices]
                    
                    # Sort the top-k
                    sort_order = np.argsort(-top_scores)
                    top_indices = top_indices[sort_order]
                    top_scores = top_scores[sort_order]
                
                self.operation_count += len(query_similarities)  # Count comparisons
                
                # Store results
                neighbors = []
                for rank, (train_idx, score) in enumerate(zip(top_indices, top_scores)):
                    if score <= 0 and exclude_self:  # Skip invalid scores
                        continue
                        
                    neighbors.append({
                        'rank': rank + 1,
                        'train_idx': int(train_idx),
                        'score': float(score),
                        'sentence': train_sentences[train_idx]
                    })
                    
                    if len(neighbors) >= self.top_k:
                        break
                
                results.append((global_query_idx, neighbors))
            
            # Memory management
            del similarities
            gc.collect()
            
            if chunk_start % (self.chunk_size * 5) == 0:  # Log every 5 chunks
                MemoryMonitor.log_memory(f"chunk {chunk_start//self.chunk_size + 1}")
        
        return results

    def save_results(self, results, query_sentences, output_path, dataset_name):
        """Save nearest neighbor results to file."""
        print(f"Saving {len(results)} results to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Nearest Neighbor Search Results - {dataset_name}\n")
            f.write(f"Query set size: {len(results)}\n")
            f.write(f"Top-K neighbors: {self.top_k}\n")
            f.write(f"Chunk size: {self.chunk_size}\n")
            f.write("="*80 + "\n\n")
            
            for query_idx, neighbors in results:
                f.write(f"Query [{query_idx}]: {query_sentences[query_idx]}\n")
                
                for neighbor in neighbors:
                    f.write(f"  Neighbor {neighbor['rank']}: ")
                    f.write(f"TrainIndex={neighbor['train_idx']:5d} ")
                    f.write(f"Score={neighbor['score']:.6f} ")
                    f.write(f"Sentence={neighbor['sentence']}\n")
                
                f.write("\n")

    def analyze_computational_complexity(self, n_train, n_val, n_test):
        """Analyze and report computational complexity."""
        
        # Training vs Training: O(n_train^2)
        train_train_ops = n_train * n_train
        
        # Validation vs Training: O(n_val * n_train) 
        val_train_ops = n_val * n_train
        
        # Test vs Training: O(n_test * n_train)
        test_train_ops = n_test * n_train
        
        total_ops = train_train_ops + val_train_ops + test_train_ops
        
        analysis = {
            'dataset_sizes': {
                'train': n_train,
                'validation': n_val, 
                'test': n_test
            },
            'operations': {
                'train_vs_train': train_train_ops,
                'val_vs_train': val_train_ops,
                'test_vs_train': test_train_ops,
                'total_operations': total_ops
            },
            'optimizations': {
                'chunk_size': self.chunk_size,
                'memory_efficient': True,
                'normalized_vectors': True,
                'sparse_matrices': True
            },
            'actual_computations': {
                'similarity_computations': self.similarity_computations,
                'comparison_operations': self.operation_count
            }
        }
        
        return analysis

    def save_operations_analysis(self, analysis, output_path):
        """Save computational analysis to file."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Computational Complexity Analysis\n")
            f.write("="*50 + "\n\n")
            
            # Dataset sizes
            f.write("Dataset Sizes:\n")
            for name, size in analysis['dataset_sizes'].items():
                f.write(f"  {name.capitalize()}: {size:,} sentences\n")
            f.write("\n")
            
            # Theoretical operations
            f.write("Theoretical Operations Count:\n")
            ops = analysis['operations']
            f.write(f"  Train vs Train: {ops['train_vs_train']:,} operations\n")
            f.write(f"  Validation vs Train: {ops['val_vs_train']:,} operations\n") 
            f.write(f"  Test vs Train: {ops['test_vs_train']:,} operations\n")
            f.write(f"  Total: {ops['total_operations']:,} operations\n\n")
            
            # Optimizations applied
            f.write("Optimizations Applied:\n")
            opts = analysis['optimizations']
            f.write(f"  Chunk-based processing: {opts['chunk_size']} samples per chunk\n")
            f.write(f"  Memory efficient: {opts['memory_efficient']}\n")
            f.write(f"  Normalized vectors: {opts['normalized_vectors']}\n") 
            f.write(f"  Sparse matrices: {opts['sparse_matrices']}\n\n")
            
            # Actual computations
            f.write("Actual Computations Performed:\n")
            actual = analysis['actual_computations']
            f.write(f"  Similarity computations: {actual['similarity_computations']:,}\n")
            f.write(f"  Comparison operations: {actual['comparison_operations']:,}\n\n")
            
            # Efficiency metrics
            theoretical_total = ops['total_operations']
            actual_total = actual['similarity_computations']
            if theoretical_total > 0:
                efficiency = (actual_total / theoretical_total) * 100
                f.write(f"Efficiency Metrics:\n")
                f.write(f"  Computation efficiency: {efficiency:.2f}%\n")
                f.write(f"  Memory chunking reduced peak memory usage\n")
                f.write(f"  Sparse matrix format reduced storage requirements\n\n")
            
            # Complexity analysis
            f.write("Computational Complexity Analysis:\n")
            f.write("  Without optimizations:\n")
            f.write("    Time: O(n²) for train-train + O(n*m) for val/test-train\n") 
            f.write("    Space: O(n²) for storing all similarities\n")
            f.write("  With optimizations:\n")
            f.write("    Time: Same complexity but with chunked processing\n")
            f.write("    Space: O(chunk_size * n) - significant memory reduction\n")
            f.write("    Additional benefits: Sparse matrices, normalized vectors\n\n")
            
            f.write("Recommendations for Large Datasets:\n")
            f.write("  1. Use approximate methods (LSH, random projections)\n")
            f.write("  2. Implement early stopping for top-k search\n")
            f.write("  3. Consider distributed computing for very large datasets\n")
            f.write("  4. Use dimensionality reduction (SVD, PCA) on TF-IDF vectors\n")

def main():
    print("Lab 7 - Bonus: Extended Nearest Neighbor Search")
    print("="*60)
    
    # Initialize
    nn_search = EfficientNearestNeighbors(chunk_size=500, top_k=5)
    MemoryMonitor.log_memory("start")
    
    # Load sentences
    print("Loading sentences...")
    train_sentences = nn_search.load_sentences(TRAIN_PATH)
    val_sentences = nn_search.load_sentences(VAL_PATH) 
    test_sentences = nn_search.load_sentences(TEST_PATH)
    
    print(f"Loaded sentences:")
    print(f"  Train: {len(train_sentences):,}")
    print(f"  Validation: {len(val_sentences):,}")
    print(f"  Test: {len(test_sentences):,}")
    
    MemoryMonitor.log_memory("sentences loaded")
    
    # Load TF-IDF matrices
    print("\nLoading TF-IDF matrices...")
    train_tfidf = nn_search.load_sparse_matrix(TFIDF_TRAIN_NPZ)
    val_tfidf = nn_search.load_sparse_matrix(TFIDF_VAL_NPZ)
    test_tfidf = nn_search.load_sparse_matrix(TFIDF_TEST_NPZ)
    
    print(f"TF-IDF matrix shapes:")
    print(f"  Train: {train_tfidf.shape}")
    print(f"  Validation: {val_tfidf.shape}")
    print(f"  Test: {test_tfidf.shape}")
    
    MemoryMonitor.log_memory("matrices loaded")
    
    # Normalize matrices for efficient cosine similarity
    print("\nNormalizing matrices...")
    train_tfidf_norm = nn_search.normalize_matrix(train_tfidf)
    val_tfidf_norm = nn_search.normalize_matrix(val_tfidf)
    test_tfidf_norm = nn_search.normalize_matrix(test_tfidf)
    
    MemoryMonitor.log_memory("matrices normalized")
    
    # Track timing
    total_start_time = time.time()
    
    # 1. Training vs Training (most computationally expensive)
    print(f"\n1. Finding nearest neighbors within training set...")
    print(f"   This involves {len(train_sentences):,} x {len(train_sentences):,} = {len(train_sentences)**2:,} comparisons")
    
    train_start_time = time.time()
    train_results = nn_search.chunked_similarity_search(
        train_tfidf_norm, train_tfidf_norm,
        train_sentences, train_sentences,
        exclude_self=True, query_start_idx=0
    )
    train_time = time.time() - train_start_time
    
    nn_search.save_results(train_results, train_sentences, TRAIN_NEIGHBORS_PATH, "Training Set")
    print(f"   Training search completed in {train_time:.2f} seconds")
    MemoryMonitor.log_memory("training search completed")
    
    # 2. Validation vs Training
    if len(val_sentences) > 0:
        print(f"\n2. Finding nearest neighbors for validation set...")
        val_start_time = time.time()
        val_results = nn_search.chunked_similarity_search(
            val_tfidf_norm, train_tfidf_norm,
            val_sentences, train_sentences,
            exclude_self=False
        )
        val_time = time.time() - val_start_time
        
        nn_search.save_results(val_results, val_sentences, VAL_NEIGHBORS_PATH, "Validation Set")
        print(f"   Validation search completed in {val_time:.2f} seconds")
        MemoryMonitor.log_memory("validation search completed")
    
    # 3. Test vs Training  
    if len(test_sentences) > 0:
        print(f"\n3. Finding nearest neighbors for test set...")
        test_start_time = time.time()
        test_results = nn_search.chunked_similarity_search(
            test_tfidf_norm, train_tfidf_norm,
            test_sentences, train_sentences,
            exclude_self=False
        )
        test_time = time.time() - test_start_time
        
        nn_search.save_results(test_results, test_sentences, TEST_NEIGHBORS_PATH, "Test Set")
        print(f"   Test search completed in {test_time:.2f} seconds")
        MemoryMonitor.log_memory("test search completed")
    
    total_time = time.time() - total_start_time
    
    # Analyze computational complexity
    print(f"\n4. Analyzing computational complexity...")
    analysis = nn_search.analyze_computational_complexity(
        len(train_sentences), len(val_sentences), len(test_sentences)
    )
    
    # Add timing information
    analysis['timing'] = {
        'train_search_time': train_time,
        'val_search_time': val_time if len(val_sentences) > 0 else 0,
        'test_search_time': test_time if len(test_sentences) > 0 else 0,
        'total_time': total_time
    }
    
    nn_search.save_operations_analysis(analysis, OPERATIONS_ANALYSIS_PATH)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total similarity computations: {nn_search.similarity_computations:,}")
    print(f"Total comparison operations: {nn_search.operation_count:,}")
    print(f"\nOutput files created:")
    print(f"  - Training neighbors: {TRAIN_NEIGHBORS_PATH}")
    if len(val_sentences) > 0:
        print(f"  - Validation neighbors: {VAL_NEIGHBORS_PATH}")
    if len(test_sentences) > 0:
        print(f"  - Test neighbors: {TEST_NEIGHBORS_PATH}")
    print(f"  - Operations analysis: {OPERATIONS_ANALYSIS_PATH}")
    
    MemoryMonitor.log_memory("completed")
    
    print(f"\nComputational Efficiency Strategies Used:")
    print(f"  ✓ Chunked processing to manage memory")
    print(f"  ✓ Matrix normalization for efficient cosine similarity")
    print(f"  ✓ Sparse matrix format for memory efficiency")
    print(f"  ✓ argpartition for efficient top-k selection")
    print(f"  ✓ Memory monitoring and garbage collection")
    print(f"  ✓ sklearn's optimized cosine_similarity for large datasets")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()