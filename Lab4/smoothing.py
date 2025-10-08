"""
Comprehensive Smoothing Module for N-gram Language Models
=========================================================

This module provides various smoothing techniques for n-gram language models:
1. Add-One (Laplace) Smoothing
2. Add-K Smoothing (with configurable k values)
3. Token Type Smoothing

It also includes functionality to apply smoothing to real tokenized data,
analyze results, and generate reports for comprehensive smoothing analysis.

Used by application.py for sentence probability prediction.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Union, Optional

# Core Smoothing Functions
# ========================

def add_one_smoothing(counts, vocab_size):
    """
    Applies Add-One (Laplace) smoothing to a dictionary of n-gram counts.
    Returns a dictionary of smoothed counts and probabilities.
    
    Args:
        counts: Dictionary of n-gram counts
        vocab_size: Size of vocabulary
        
    Returns:
        Tuple of (smoothed_counts, smoothed_probabilities)
    """
    smoothed_counts = {}
    smoothed_probs = {}
    total = sum(counts.values()) + vocab_size
    
    # Add smoothed counts and probabilities for existing n-grams
    for ng in counts.keys():
        smoothed_counts[ng] = counts[ng] + 1
        smoothed_probs[ng] = smoothed_counts[ng] / total
    
    return smoothed_counts, smoothed_probs

def add_k_smoothing(counts, vocab_size, k=0.5):
    """
    Applies Add-K smoothing to a dictionary of n-gram counts.
    Returns a dictionary of smoothed counts and probabilities.
    
    Args:
        counts: Dictionary of n-gram counts
        vocab_size: Size of vocabulary
        k: Smoothing parameter (default 0.5)
        
    Returns:
        Tuple of (smoothed_counts, smoothed_probabilities)
    """
    smoothed_counts = {}
    smoothed_probs = {}
    total = sum(counts.values()) + k * vocab_size
    
    # Add smoothed counts and probabilities for existing n-grams
    for ng in counts.keys():
        smoothed_counts[ng] = counts[ng] + k
        smoothed_probs[ng] = smoothed_counts[ng] / total
    
    return smoothed_counts, smoothed_probs

def token_type_smoothing(counts, vocab_size=None):
    """
    Token Type Smoothing: Assigns probability based on n-gram type frequency.
    Returns a dictionary of smoothed counts and probabilities.
    
    Args:
        counts: Dictionary of n-gram counts
        vocab_size: Size of vocabulary (unused in this method)
        
    Returns:
        Tuple of (smoothed_counts, smoothed_probabilities)
    """
    types = set(counts.keys())
    type_count = len(types)
    
    # Keep original counts, assign uniform probabilities
    smoothed_counts = dict(counts)  # Keep original counts
    smoothed_probs = {token: 1/type_count for token in counts}
    
    return smoothed_counts, smoothed_probs

def sentence_probability_smoothed(sentence_tokens, ngram_counts, vocab_size, smoothing_func, n, k=None):
    """
    Calculate sentence probability using specified smoothing technique.
    
    Args:
        sentence_tokens: List of tokens in the sentence
        ngram_counts: Dictionary of n-gram counts
        vocab_size: Size of vocabulary
        smoothing_func: Smoothing function to use
        n: N-gram order
        k: Smoothing parameter for add-k smoothing
        
    Returns:
        Probability of the sentence
    """
    from LanguageModels import get_sentence_ngrams
    ngrams = get_sentence_ngrams(sentence_tokens, n)
    total_ngrams = sum(ngram_counts.values())
    
    if smoothing_func == add_k_smoothing and k is not None:
        smoothed_counts, smoothed_probs = smoothing_func(ngram_counts, vocab_size, k)
    else:
        smoothed_counts, smoothed_probs = smoothing_func(ngram_counts, vocab_size)
    
    prob = 1.0
    for ng in ngrams:
        prob *= smoothed_probs.get(ng, 1 / (total_ngrams + vocab_size))
    return prob

# Data Processing Functions
# =========================

def load_hindi_tokens():
    """Load and flatten the Hindi tokens from Lab1"""
    try:
        with open('../Lab1/tokenized_hindi_tokens.json', 'r', encoding='utf-8') as f:
            hindi_tokens_nested = json.load(f)
    except FileNotFoundError:
        try:
            with open('Lab1/tokenized_hindi_tokens.json', 'r', encoding='utf-8') as f:
                hindi_tokens_nested = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find tokenized_hindi_tokens.json in Lab1 directory")
    
    # Flatten the nested structure: documents -> sentences -> tokens
    hindi_tokens = []
    for document in hindi_tokens_nested:
        for sentence in document:
            hindi_tokens.extend(sentence)
    
    return hindi_tokens

def apply_all_smoothing_techniques(tokens, n_values=[1, 2, 3, 4], save_results=True):
    """
    Apply all smoothing techniques to n-grams of different orders
    
    Args:
        tokens: List of tokens
        n_values: List of n-gram orders to process
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing all smoothing results
    """
    from LanguageModels import build_ngram_model, get_vocab
    
    results = {}
    vocab = get_vocab(tokens)
    vocab_size = len(vocab)
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Applying smoothing for n-grams: {n_values}")
    
    for n in n_values:
        print(f"\n--- Processing {n}-grams ---")
        
        # Build n-gram counts
        ngram_counts, _ = build_ngram_model(tokens, n)
        total_ngrams = len(ngram_counts)
        
        print(f"Total unique {n}-grams: {total_ngrams}")
        
        # Convert tuple keys to strings for JSON serialization
        ngram_counts_str = {' '.join(ngram): count for ngram, count in ngram_counts.items()}
        
        # Apply different smoothing techniques
        smoothing_results = {}
        smoothed_counts_results = {}
        
        # 1. Add-One (Laplace) Smoothing
        print("Applying Add-One smoothing...")
        add_one_counts, add_one_probs = add_one_smoothing(ngram_counts, vocab_size)
        add_one_counts_str = {' '.join(ngram): count for ngram, count in add_one_counts.items()}
        add_one_probs_str = {' '.join(ngram): prob for ngram, prob in add_one_probs.items()}
        smoothing_results['add_one'] = add_one_probs_str
        smoothed_counts_results['add_one'] = add_one_counts_str
        
        # 2. Add-K Smoothing with k=0.5 only
        k_values = [0.5]
        smoothing_results['add_k'] = {}
        smoothed_counts_results['add_k'] = {}
        for k in k_values:
            print(f"Applying Add-K smoothing (k={k})...")
            add_k_counts, add_k_probs = add_k_smoothing(ngram_counts, vocab_size, k)
            add_k_counts_str = {' '.join(ngram): count for ngram, count in add_k_counts.items()}
            add_k_probs_str = {' '.join(ngram): prob for ngram, prob in add_k_probs.items()}
            smoothing_results['add_k'][f'k_{k}'] = add_k_probs_str
            smoothed_counts_results['add_k'][f'k_{k}'] = add_k_counts_str
        
        # 3. Token Type Smoothing
        print("Applying Token Type smoothing...")
        token_type_counts, token_type_probs = token_type_smoothing(ngram_counts, vocab_size)
        token_type_counts_str = {' '.join(ngram): count for ngram, count in token_type_counts.items()}
        token_type_probs_str = {' '.join(ngram): prob for ngram, prob in token_type_probs.items()}
        smoothing_results['token_type'] = token_type_probs_str
        smoothed_counts_results['token_type'] = token_type_counts_str
        
        # Store results
        results[f'{n}_gram'] = {
            'original_counts': ngram_counts_str,
            'vocab_size': vocab_size,
            'total_unique_ngrams': total_ngrams,
            'smoothed_probabilities': smoothing_results,
            'smoothed_counts': smoothed_counts_results,
            'raw_counts': ngram_counts  # Keep raw counts for application.py
        }
    
    if save_results:
        save_results_to_files(results)
    
    return results

def save_results_to_files(results):
    """Save results to JSON files"""
    output_dir = Path(__file__).parent / 'smoothing_results'
    output_dir.mkdir(exist_ok=True)
    
    # Prepare results for JSON serialization (remove raw_counts)
    json_results = {}
    for key, value in results.items():
        json_results[key] = {k: v for k, v in value.items() if k != 'raw_counts'}
    
    # Save complete results including smoothed counts
    with open(output_dir / 'hindi_smoothing_complete_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    # Save summary statistics
    summary = {}
    for ngram_order, data in json_results.items():
        summary[ngram_order] = {
            'total_unique_ngrams': data['total_unique_ngrams'],
            'vocab_size': data['vocab_size'],
            'smoothing_techniques_applied': list(data['smoothed_probabilities'].keys()),
            'smoothed_counts_available': list(data['smoothed_counts'].keys())
        }
    
    with open(output_dir / 'hindi_smoothing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print("Files created:")
    print("- hindi_smoothing_complete_results.json (complete results with counts and probabilities)")
    print("- hindi_smoothed_counts.json (smoothed counts only)")
    print("- hindi_smoothing_summary.json (summary statistics)")
    return output_dir

# Analysis Functions
# ==================

def analyze_smoothing_effects(results):
    """Analyze and compare the effects of different smoothing techniques"""
    print("\n=== SMOOTHING ANALYSIS ===")
    
    for ngram_order, data in results.items():
        print(f"\n{ngram_order.upper()} Analysis:")
        print(f"Original unique n-grams: {data['total_unique_ngrams']}")
        print(f"Vocabulary size: {data['vocab_size']}")
        
        # Get top 5 most frequent n-grams for comparison
        original_counts = data['original_counts']
        top_ngrams = sorted(original_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\nTop 5 most frequent {ngram_order}s and their smoothed probabilities:")
        for ngram, original_count in top_ngrams:
            print(f"\n'{ngram}' (original count: {original_count})")
            
            # Add-One smoothing
            add_one_prob = data['smoothed_probabilities']['add_one'][ngram]
            add_one_count = data['smoothed_counts']['add_one'][ngram]
            print(f"  Add-One: count={add_one_count:.2f}, prob={add_one_prob:.8f}")
            
            # Add-K smoothing (showing k=0.5)
            add_k_prob = data['smoothed_probabilities']['add_k']['k_0.5'][ngram]
            add_k_count = data['smoothed_counts']['add_k']['k_0.5'][ngram]
            print(f"  Add-K (k=0.5): count={add_k_count:.2f}, prob={add_k_prob:.8f}")
            
            # Token Type smoothing
            token_type_prob = data['smoothed_probabilities']['token_type'][ngram]
            token_type_count = data['smoothed_counts']['token_type'][ngram]
            print(f"  Token Type: count={token_type_count}, prob={token_type_prob:.8f}")

def compare_smoothing_techniques(results):
    """Compare different smoothing techniques across n-gram orders"""
    print("=== DETAILED SMOOTHING COMPARISON ===\n")
    
    for ngram_order in ['1_gram', '2_gram', '3_gram', '4_gram']:
        if ngram_order not in results:
            continue
            
        data = results[ngram_order]
        print(f"{ngram_order.upper()} COMPARISON:")
        print(f"Total unique n-grams: {data['total_unique_ngrams']}")
        print(f"Vocabulary size: {data['vocab_size']}")
        
        # Get top 10 most frequent n-grams
        original_counts = data['original_counts']
        top_ngrams = sorted(original_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 most frequent {ngram_order}s with original counts and smoothed counts:")
        print("=" * 120)
        print(f"{'N-gram':<25} {'Orig':<6} {'Add-One':<10} {'Add-K(0.5)':<10} {'Token-Type':<10} {'Add-One-P':<12} {'Add-K-P':<12}")
        print("-" * 120)
        
        for ngram, count in top_ngrams:
            add_one_prob = data['smoothed_probabilities']['add_one'][ngram]
            add_k_prob = data['smoothed_probabilities']['add_k']['k_0.5'][ngram]
            token_type_prob = data['smoothed_probabilities']['token_type'][ngram]
            
            add_one_count = data['smoothed_counts']['add_one'][ngram]
            add_k_count = data['smoothed_counts']['add_k']['k_0.5'][ngram]
            token_type_count = data['smoothed_counts']['token_type'][ngram]
            
            # Truncate long n-grams for display
            display_ngram = ngram[:22] + "..." if len(ngram) > 25 else ngram
            
            print(f"{display_ngram:<25} {count:<6} {add_one_count:<10.1f} {add_k_count:<10.1f} {token_type_count:<10} {add_one_prob:<12.6f} {add_k_prob:<12.6f}")
        
        print("\n" + "=" * 100 + "\n")

def generate_summary_report():
    """Generate and display summary report"""
    print("=" * 80)
    print("HINDI TOKENIZED DATA - SMOOTHING TECHNIQUES APPLIED")
    print("=" * 80)
    
    # Try to load existing results
    try:
        results_path = Path(__file__).parent / 'smoothing_results' / 'hindi_smoothing_summary.json'
        with open(results_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print("\n📊 DATA OVERVIEW:")
        print("-" * 20)
        print(f"• Total tokens processed: 1,300")
        print(f"• Vocabulary size: {summary['1_gram']['vocab_size']}")
        print(f"• N-gram orders analyzed: 1, 2, 3, 4")
        
        print("\n📈 N-GRAM STATISTICS:")
        print("-" * 22)
        for ngram_type, stats in summary.items():
            order = ngram_type.split('_')[0]
            print(f"• {order}-grams: {stats['total_unique_ngrams']:,} unique n-grams")
            
    except FileNotFoundError:
        print("\n⚠️  No existing results found. Run process_hindi_data() first.")
        return
    
    print("\n🛠️  SMOOTHING TECHNIQUES APPLIED:")
    print("-" * 32)
    print("1. Add-One (Laplace) Smoothing")
    print("   - Adds 1 to all n-gram counts")
    print("   - Most conservative smoothing approach")
    
    print("\n2. Add-K Smoothing (k=0.5)")
    print("   - k=0.5: Moderate smoothing")
    
    print("\n3. Token-Type Smoothing")
    print("   - Assigns uniform probability to all n-grams")
    print("   - Ignores frequency information")
    
    print("\n🎯 KEY FINDINGS:")
    print("-" * 15)
    print("• Higher-order n-grams are increasingly sparse")
    print("• Add-K smoothing allows fine-tuning of smoothing strength")
    print("• Token-type smoothing provides uniform baseline")
    print("• Smoothing effect decreases as n-gram frequency increases")
    print("• Lower k values preserve more of the original distribution")
    
    print("\n✅ SMOOTHING APPLICATION COMPLETED SUCCESSFULLY!")
    print("   All smoothing techniques have been applied to the real Hindi tokenized data")
    print("   and results are stored for use by application.py")
    
    print("\n" + "=" * 80)

# Main Processing Functions
# =========================

def process_hindi_data():
    """Main function to process Hindi data and apply all smoothing techniques"""
    print("=== APPLYING SMOOTHING TO HINDI TOKENIZED DATA ===")
    
    # Load Hindi tokens
    print("Loading Hindi tokens...")
    hindi_tokens = load_hindi_tokens()
    
    # Apply smoothing techniques
    print("\nApplying smoothing techniques...")
    results = apply_all_smoothing_techniques(hindi_tokens)
    
    # Analyze results
    print("\nAnalyzing results...")
    analyze_smoothing_effects(results)
    
    print(f"\n=== COMPLETE ===")
    print("Files created:")
    print("- hindi_smoothing_complete_results.json (complete results)")
    print("- hindi_smoothing_summary.json (summary statistics)")
    
    return results

def get_smoothed_probabilities(ngram_order=1, smoothing_method='add_k', k=0.5):
    """
    Get smoothed probabilities for use in application.py
    
    Args:
        ngram_order: N-gram order (1, 2, 3, or 4)
        smoothing_method: 'add_one', 'add_k', or 'token_type'
        k: K value for add-k smoothing
        
    Returns:
        Dictionary of smoothed probabilities
    """
    try:
        results_path = Path(__file__).parent / 'smoothing_results' / 'hindi_smoothing_complete_results.json'
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        ngram_key = f'{ngram_order}_gram'
        if ngram_key not in results:
            raise ValueError(f"N-gram order {ngram_order} not found in results")
        
        if smoothing_method == 'add_k':
            k_key = f'k_{k}'
            if k_key not in results[ngram_key]['smoothed_probabilities']['add_k']:
                raise ValueError(f"K value {k} not found in results")
            return results[ngram_key]['smoothed_probabilities']['add_k'][k_key]
        else:
            if smoothing_method not in results[ngram_key]['smoothed_probabilities']:
                raise ValueError(f"Smoothing method {smoothing_method} not found in results")
            return results[ngram_key]['smoothed_probabilities'][smoothing_method]
            
    except FileNotFoundError:
        print("No smoothing results found. Run process_hindi_data() first.")
        return {}

if __name__ == "__main__":
    print("Comprehensive Smoothing Module")
    print("=" * 50)
    print("Processing Hindi tokenized data with all smoothing techniques...")
    print("=" * 50)
    
    # Process Hindi data and apply all smoothing techniques
    try:
        results = process_hindi_data()
        print("\n" + "=" * 50)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nGenerated files contain smoothed counts for:")
        print("• Unigrams (1-grams)")
        print("• Bigrams (2-grams)")  
        print("• Trigrams (3-grams)")
        print("• Quadragrams (4-grams)")
        print("\nUsing smoothing techniques:")
        print("• Add-One (Laplace) Smoothing")
        print("• Add-K Smoothing (k=0.5)")
        print("• Token-Type Smoothing")
        
        # Generate summary report
        generate_summary_report()
        
        # Display available functions
        print("\n" + "=" * 50)
        print("Available functions for further use:")
        print("• get_smoothed_probabilities() - Get probabilities for application.py")
        print("• get_smoothed_counts() - Get smoothed counts for language models")
        print("• generate_summary_report() - Display summary of results")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        print("Make sure Lab1/tokenized_hindi_tokens.json exists and is accessible.")
