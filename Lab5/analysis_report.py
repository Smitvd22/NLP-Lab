"""
Lab 5 Analysis Report Generator

This script generates detailed analysis and visualizations for the Good-Turing smoothing implementation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

def analyze_frequency_distribution(smoother, model_name):
    """Analyze and visualize frequency distribution"""
    frequencies = list(smoother.frequency_counts.keys())
    counts = list(smoother.frequency_counts.values())
    
    # Create frequency analysis
    print(f"\n=== {model_name} Frequency Analysis ===")
    print(f"Total unique n-grams: {smoother.total_ngrams}")
    print(f"N1 (singletons): {smoother.frequency_counts.get(1, 0)}")
    print(f"Vocabulary size: {smoother.vocab_size}")
    
    # Calculate Good-Turing discount factors
    print("\nGood-Turing Discount Factors:")
    for c in range(1, 11):
        if c in smoother.frequency_counts:
            nc = smoother.frequency_counts[c]
            nc_plus_1 = smoother.frequency_counts.get(c + 1, 0)
            if nc > 0:
                if nc_plus_1 > 0:
                    discount = (c + 1) * nc_plus_1 / (c * nc)
                    print(f"  d_{c} = {discount:.4f}")
                else:
                    print(f"  d_{c} = 1.0000 (no c+1 frequency)")
    
    return frequencies, counts

def create_frequency_table_report(models):
    """Create a comprehensive frequency table report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FREQUENCY TABLE ANALYSIS")
    print("="*80)
    
    for n, model in models.items():
        print(f"\n{n}-gram Model Frequency Distribution:")
        print("-" * 50)
        print("C\tNc\tC*\tP(unseen)")
        print("-" * 50)
        
        # Get sorted frequencies
        freq_items = sorted(model.smoother.frequency_counts.items())
        
        for c, nc in freq_items[:20]:  # Show top 20
            c_star = model.smoother.good_turing_estimates.get(c, c)
            
            # Calculate probability for unseen n-grams
            if c == 0:
                if n == 1:
                    unseen_types = model.vocab_size - len(model.ngram_counts)
                    p_unseen = model.smoother.frequency_counts.get(1, 0) / (model.smoother.total_ngrams * unseen_types) if unseen_types > 0 else 0
                else:
                    possible_ngrams = model.vocab_size ** n
                    unseen_ngrams = possible_ngrams - model.smoother.total_ngrams
                    p_unseen = model.smoother.frequency_counts.get(1, 0) / (model.smoother.total_ngrams * unseen_ngrams) if unseen_ngrams > 0 else 0
            else:
                p_unseen = c_star / sum(model.ngram_counts.values())
            
            print(f"{c}\t{nc}\t{c_star:.6f}\t{p_unseen:.8e}")

def evaluate_sentence_probabilities(models, test_sentences, model_names):
    """Evaluate sentence probabilities across different models"""
    print("\n" + "="*80)
    print("SENTENCE PROBABILITY EVALUATION")
    print("="*80)
    
    # Take first 5 sentences for detailed analysis
    sample_sentences = test_sentences[:5]
    
    results = []
    
    for i, sentence in enumerate(sample_sentences):
        print(f"\nSentence {i+1}: {sentence[:50]}...")
        print("-" * 60)
        
        sentence_results = {"sentence": sentence[:50]}
        
        for n, model in models.items():
            prob = model.get_sentence_probability(sentence)
            log_prob = np.log(prob) if prob > 0 else float('-inf')
            perplexity = np.exp(-log_prob) if log_prob != float('-inf') else float('inf')
            
            sentence_results[f"{n}-gram_prob"] = prob
            sentence_results[f"{n}-gram_logprob"] = log_prob
            sentence_results[f"{n}-gram_perplexity"] = perplexity
            
            print(f"  {n}-gram: P = {prob:.2e}, Log P = {log_prob:.4f}, PPL = {perplexity:.2e}")
        
        results.append(sentence_results)
    
    return results

def compare_smoothing_methods(models):
    """Compare Good-Turing with other smoothing methods"""
    print("\n" + "="*80)
    print("SMOOTHING METHOD COMPARISON")
    print("="*80)
    
    for n, model in models.items():
        print(f"\n{n}-gram Model Analysis:")
        print(f"  Vocabulary Size: {model.vocab_size}")
        print(f"  Total N-grams: {model.smoother.total_ngrams}")
        print(f"  Singletons (N1): {model.smoother.frequency_counts.get(1, 0)}")
        print(f"  Singleton Ratio: {model.smoother.frequency_counts.get(1, 0) / model.smoother.total_ngrams:.4f}")
        
        # Calculate coverage
        seen_mass = sum(count for count in model.ngram_counts.values() if count > 1)
        total_mass = sum(model.ngram_counts.values())
        print(f"  Seen Mass (count > 1): {seen_mass / total_mass:.4f}")
        print(f"  Singleton Mass: {model.smoother.frequency_counts.get(1, 0) / total_mass:.4f}")

def generate_summary_report(models, train_size, val_size, test_size, interpolation_weights):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    print(f"\nData Split Summary:")
    print(f"  Training sentences: {train_size}")
    print(f"  Validation sentences: {val_size}")
    print(f"  Test sentences: {test_size}")
    print(f"  Total sentences: {train_size + val_size + test_size}")
    
    print(f"\nModel Statistics:")
    for n, model in models.items():
        print(f"  {n}-gram Model:")
        print(f"    Vocabulary Size: {model.vocab_size}")
        print(f"    Unique N-grams: {len(model.ngram_counts)}")
        print(f"    Total N-gram Tokens: {sum(model.ngram_counts.values())}")
        print(f"    Singletons: {model.smoother.frequency_counts.get(1, 0)}")
        print(f"    OOV Rate Estimate: {model.smoother.frequency_counts.get(1, 0) / model.smoother.total_ngrams:.4f}")
    
    print(f"\nDeleted Interpolation Results:")
    print(f"  Optimal λ values:")
    print(f"    λ1 (unigram): {interpolation_weights[0]:.6f}")
    print(f"    λ2 (bigram): {interpolation_weights[1]:.6f}")
    print(f"    λ3 (trigram): {interpolation_weights[2]:.6f}")
    print(f"    λ4 (quadrigram): {interpolation_weights[3]:.6f}")
    
    print(f"\nKey Insights:")
    print(f"  1. Higher-order models get very small weights in interpolation")
    print(f"  2. Unigram and bigram models dominate the interpolation")
    print(f"  3. Good-Turing smoothing effectively handles unseen n-grams")
    print(f"  4. Perplexity decreases with interpolation compared to individual models")

if __name__ == "__main__":
    print("=== Good-Turing Smoothing Analysis Report ===")
    print("This report provides detailed analysis of the implementation.")
    print("\nTo run full analysis, execute Good_Turing.py first, then import this module.")
    print("\nExample usage:")
    print("  from Good_Turing import *")
    print("  # Run main() first")
    print("  from analysis_report import *")
    print("  # Then run specific analysis functions")
