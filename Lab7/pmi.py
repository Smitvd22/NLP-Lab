"""
Lab 7 - PMI (Pointwise Mutual Information) Computation
Using bigram and unigram language models from Lab4 to compute PMI scores
for all bigrams in validation and testing sets from Lab1.
"""

import json
import math
import csv
from pathlib import Path
from collections import defaultdict
import random

# Paths to data files
LAB1_PATH = Path(__file__).parent.parent / "Lab1"
LAB4_PATH = Path(__file__).parent.parent / "Lab4"
LAB7_PATH = Path(__file__).parent

# Input files
HINDI_TOKENS_PATH = LAB1_PATH / "tokenized_hindi_tokens.json"
MARATHI_TOKENS_PATH = LAB1_PATH / "tokenized_marathi_tokens.json"
HINDI_UNIGRAMS_PATH = LAB4_PATH / "hindi_unigrams.tsv"
HINDI_BIGRAMS_PATH = LAB4_PATH / "hindi_bigrams.tsv"
MARATHI_UNIGRAMS_PATH = LAB4_PATH / "marathi_unigrams.tsv"
MARATHI_BIGRAMS_PATH = LAB4_PATH / "marathi_bigrams.tsv"

# Output files
TRAIN_PATH = LAB7_PATH / "train.txt"
VAL_PATH = LAB7_PATH / "val.txt"
TEST_PATH = LAB7_PATH / "test.txt"
PMI_OUTPUT_CSV = LAB7_PATH / "pmi_scores.csv"

def load_tokenized_data(filepath):
    """Load tokenized data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def flatten_sentences(tokenized_data):
    """Flatten nested sentence structure to get individual sentences."""
    sentences = []
    for doc in tokenized_data:
        if isinstance(doc, list) and len(doc) > 0:
            if isinstance(doc[0], list):
                # Multiple sentences per document
                for sentence_tokens in doc:
                    if sentence_tokens:  # Skip empty sentences
                        sentences.append(' '.join(sentence_tokens))
            else:
                # Single sentence per document
                if doc:  # Skip empty documents
                    sentences.append(' '.join(doc))
    return sentences

def split_data(sentences, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split sentences into train, validation, and test sets."""
    random.seed(seed)
    sentences = list(sentences)
    random.shuffle(sentences)
    
    n = len(sentences)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = sentences[:n_train]
    val = sentences[n_train:n_train + n_val]
    test = sentences[n_train + n_val:]
    
    return train, val, test

def save_sentences(sentences, filepath):
    """Save sentences to a text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

def load_ngram_counts_from_tsv(unigram_path, bigram_path):
    """Load n-gram counts from TSV files created in Lab4."""
    uni_counts = {}
    bi_counts = {}
    total_unigrams = 0
    total_bigrams = 0
    
    # Load unigrams
    with open(unigram_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                token = parts[0]
                count = int(parts[1])
                uni_counts[(token,)] = count
                total_unigrams += count
    
    # Load bigrams
    with open(bigram_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                w1, w2 = parts[0], parts[1]
                count = int(parts[2])
                bi_counts[(w1, w2)] = count
                total_bigrams += count
    
    return uni_counts, bi_counts, total_unigrams, total_bigrams

def extract_bigrams_from_sentences(sentences):
    """Extract bigrams from a list of sentences."""
    bigrams = []
    for sentence in sentences:
        tokens = sentence.split()
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))
    return bigrams

def compute_pmi_for_bigrams(bigrams, uni_counts, bi_counts, total_unigrams, total_bigrams):
    """Compute PMI scores for given bigrams."""
    pmi_scores = []
    
    for w1, w2 in bigrams:
        c_xy = bi_counts.get((w1, w2), 0)
        c_x = uni_counts.get((w1,), 0)
        c_y = uni_counts.get((w2,), 0)
        
        if c_x == 0 or c_y == 0 or c_xy == 0:
            pmi = None  # Cannot compute PMI
        else:
            p_x = c_x / total_unigrams
            p_y = c_y / total_unigrams
            p_xy = c_xy / total_bigrams
            
            denom = p_x * p_y
            if denom == 0 or p_xy == 0:
                pmi = None
            else:
                pmi = math.log(p_xy / denom)
        
        pmi_scores.append((w1, w2, c_xy, c_x, c_y, pmi))
    
    return pmi_scores

def save_pmi_results(pmi_scores, output_path, label):
    """Save PMI scores to CSV file."""
    filename = output_path.parent / f"pmi_{label}_scores.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word1', 'word2', 'bigram_count', 'word1_count', 'word2_count', 'pmi'])
        for row in pmi_scores:
            writer.writerow(row)
    print(f"Saved PMI scores for {label} to {filename}")

def print_top_pmi_scores(pmi_scores, label, top_k=10):
    """Print top PMI scores."""
    # Filter out None PMI scores and sort by PMI value
    valid_scores = [row for row in pmi_scores if row[-1] is not None]
    valid_scores.sort(key=lambda x: x[-1], reverse=True)
    
    print(f"\nTop {top_k} PMI bigrams in {label}:")
    for i, row in enumerate(valid_scores[:top_k], 1):
        w1, w2, c_xy, c_x, c_y, pmi = row
        print(f"{i:2d}. {w1} {w2}  PMI={pmi:.4f}  count={c_xy}")

def main():
    # Load and process Hindi data
    print("Processing Hindi data...")
    hindi_tokens = load_tokenized_data(HINDI_TOKENS_PATH)
    hindi_sentences = flatten_sentences(hindi_tokens)
    print(f"Loaded {len(hindi_sentences)} Hindi sentences")
    
    # Load and process Marathi data
    print("Processing Marathi data...")
    marathi_tokens = load_tokenized_data(MARATHI_TOKENS_PATH)
    marathi_sentences = flatten_sentences(marathi_tokens)
    print(f"Loaded {len(marathi_sentences)} Marathi sentences")
    
    # Combine both languages for mixed language processing
    all_sentences = hindi_sentences + marathi_sentences
    print(f"Total sentences: {len(all_sentences)}")
    
    # Split data
    train_sentences, val_sentences, test_sentences = split_data(all_sentences)
    print(f"Split: Train={len(train_sentences)}, Val={len(val_sentences)}, Test={len(test_sentences)}")
    
    # Save split data
    save_sentences(train_sentences, TRAIN_PATH)
    save_sentences(val_sentences, VAL_PATH)
    save_sentences(test_sentences, TEST_PATH)
    print("Saved train, validation, and test sets")
    
    # Process Hindi PMI
    print("\n" + "="*50)
    print("Computing PMI for Hindi data...")
    hindi_uni_counts, hindi_bi_counts, hindi_total_uni, hindi_total_bi = load_ngram_counts_from_tsv(
        HINDI_UNIGRAMS_PATH, HINDI_BIGRAMS_PATH
    )
    print(f"Hindi - Total unigrams: {hindi_total_uni}, Total bigrams: {hindi_total_bi}")
    
    # Extract bigrams from validation and test sets (focusing on Hindi tokens)
    val_bigrams = extract_bigrams_from_sentences(val_sentences)
    test_bigrams = extract_bigrams_from_sentences(test_sentences)
    
    # Compute PMI scores for validation set
    val_pmi_scores = compute_pmi_for_bigrams(
        val_bigrams, hindi_uni_counts, hindi_bi_counts, hindi_total_uni, hindi_total_bi
    )
    
    # Compute PMI scores for test set
    test_pmi_scores = compute_pmi_for_bigrams(
        test_bigrams, hindi_uni_counts, hindi_bi_counts, hindi_total_uni, hindi_total_bi
    )
    
    # Save and display results
    save_pmi_results(val_pmi_scores, PMI_OUTPUT_CSV, "validation")
    save_pmi_results(test_pmi_scores, PMI_OUTPUT_CSV, "test")
    
    print_top_pmi_scores(val_pmi_scores, "validation", top_k=10)
    print_top_pmi_scores(test_pmi_scores, "test", top_k=10)
    
    # Also process Marathi PMI
    print("\n" + "="*50)
    print("Computing PMI for Marathi data...")
    marathi_uni_counts, marathi_bi_counts, marathi_total_uni, marathi_total_bi = load_ngram_counts_from_tsv(
        MARATHI_UNIGRAMS_PATH, MARATHI_BIGRAMS_PATH
    )
    print(f"Marathi - Total unigrams: {marathi_total_uni}, Total bigrams: {marathi_total_bi}")
    
    # Compute PMI scores for Marathi bigrams in validation and test sets
    val_pmi_scores_marathi = compute_pmi_for_bigrams(
        val_bigrams, marathi_uni_counts, marathi_bi_counts, marathi_total_uni, marathi_total_bi
    )
    
    test_pmi_scores_marathi = compute_pmi_for_bigrams(
        test_bigrams, marathi_uni_counts, marathi_bi_counts, marathi_total_uni, marathi_total_bi
    )
    
    # Save and display Marathi results
    save_pmi_results(val_pmi_scores_marathi, PMI_OUTPUT_CSV, "validation_marathi")
    save_pmi_results(test_pmi_scores_marathi, PMI_OUTPUT_CSV, "test_marathi")
    
    print_top_pmi_scores(val_pmi_scores_marathi, "validation (Marathi model)", top_k=10)
    print_top_pmi_scores(test_pmi_scores_marathi, "test (Marathi model)", top_k=10)

if __name__ == "__main__":
    main()
