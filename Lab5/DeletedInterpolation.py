"""
Lab 5 - Deleted Interpolation Smoothing Implementation

This module implements:
1. Deleted interpolation smoothing for quadrigram model
2. EM algorithm for optimizing interpolation weights
3. Combination of multiple n-gram models for better probability estimation
4. Model evaluation and comparison

Deleted Interpolation Formula:
P(w_i | w_{i-3}, w_{i-2}, w_{i-1}) = λ1*P(w_i) + λ2*P(w_i|w_{i-1}) + 
                                     λ3*P(w_i|w_{i-2}, w_{i-1}) + 
                                     λ4*P(w_i|w_{i-3}, w_{i-2}, w_{i-1})

Where λ1 + λ2 + λ3 + λ4 = 1
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import sys
import os

# Import Good-Turing implementation
try:
    from Good_Turing import (
        load_sentences, create_data_splits, build_ngram_model, 
        get_sentence_ngrams, GoodTuringSmoothing, NGramModel,
        evaluate_models_on_dataset
    )
except ImportError:
    print("Error: Could not import from Good_Turing.py. Make sure the file exists in the same directory.")
    sys.exit(1)

class DeletedInterpolation:
    """
    Deleted interpolation smoothing for quadrigram model
    
    Uses EM algorithm to find optimal lambda parameters:
    P(w_i | w_{i-3}, w_{i-2}, w_{i-1}) = λ1*P(w_i) + λ2*P(w_i|w_{i-1}) + 
                                         λ3*P(w_i|w_{i-2}, w_{i-1}) + 
                                         λ4*P(w_i|w_{i-3}, w_{i-2}, w_{i-1})
    """
    
    def __init__(self, unigram_model, bigram_model, trigram_model, quadrigram_model):
        self.models = [unigram_model, bigram_model, trigram_model, quadrigram_model]
        self.lambdas = [0.25, 0.25, 0.25, 0.25]  # Initial equal weights
        
    def optimize_lambdas(self, held_out_sentences, iterations=10):
        """Optimize lambda parameters using EM algorithm on held-out data"""
        print("Optimizing interpolation parameters using EM algorithm...")
        
        for iteration in range(iterations):
            lambda_numerators = [0.0, 0.0, 0.0, 0.0]
            lambda_denominator = 0.0
            
            for sentence in held_out_sentences:
                tokens = sentence.strip().split()
                if len(tokens) < 4:
                    continue
                    
                quadrigrams = get_sentence_ngrams(tokens, 4)
                
                for quad in quadrigrams:
                    # Get probabilities from each model
                    unigram = (quad[3],)
                    bigram = (quad[2], quad[3])
                    trigram = (quad[1], quad[2], quad[3])
                    
                    probs = [
                        self.models[0].smoother.get_smoothed_probability(unigram, self.models[0].ngram_counts, 1),
                        self.models[1].smoother.get_smoothed_probability(bigram, self.models[1].ngram_counts, 2),
                        self.models[2].smoother.get_smoothed_probability(trigram, self.models[2].ngram_counts, 3),
                        self.models[3].smoother.get_smoothed_probability(quad, self.models[3].ngram_counts, 4)
                    ]
                    
                    # Calculate interpolated probability
                    interpolated_prob = sum(self.lambdas[i] * probs[i] for i in range(4))
                    
                    if interpolated_prob > 0:
                        # E-step: calculate expected counts
                        for i in range(4):
                            weight = (self.lambdas[i] * probs[i]) / interpolated_prob
                            lambda_numerators[i] += weight
                            lambda_denominator += weight
            
            # M-step: update lambdas
            if lambda_denominator > 0:
                self.lambdas = [num / lambda_denominator for num in lambda_numerators]
                        
            print(f"Iteration {iteration + 1}: λ = [{', '.join(f'{l:.6f}' for l in self.lambdas)}]")
    
    def get_interpolated_probability(self, quadrigram):
        """Get interpolated probability for a quadrigram"""
        unigram = (quadrigram[3],)
        bigram = (quadrigram[2], quadrigram[3])
        trigram = (quadrigram[1], quadrigram[2], quadrigram[3])
        
        prob_1 = self.models[0].smoother.get_smoothed_probability(
            unigram, self.models[0].ngram_counts, 1
        )
        prob_2 = self.models[1].smoother.get_smoothed_probability(
            bigram, self.models[1].ngram_counts, 2
        )
        prob_3 = self.models[2].smoother.get_smoothed_probability(
            trigram, self.models[2].ngram_counts, 3
        )
        prob_4 = self.models[3].smoother.get_smoothed_probability(
            quadrigram, self.models[3].ngram_counts, 4
        )
        
        return (self.lambdas[0] * prob_1 + 
                self.lambdas[1] * prob_2 + 
                self.lambdas[2] * prob_3 + 
                self.lambdas[3] * prob_4)
    
    def get_sentence_probability(self, sentence):
        """Calculate sentence probability using deleted interpolation"""
        tokens = sentence.strip().split()
        if len(tokens) < 4:
            return 1e-10  # Very small probability for short sentences
            
        quadrigrams = get_sentence_ngrams(tokens, 4)
        
        prob = 1.0
        for quad in quadrigrams:
            quad_prob = self.get_interpolated_probability(quad)
            prob *= max(quad_prob, 1e-10)  # Avoid zero probabilities
            
        return prob

def train_individual_models(train_sentences):
    """Train individual n-gram models for interpolation"""
    print("Training individual n-gram models for interpolation...")
    
    models = {}
    
    for n in range(1, 5):
        print(f"Training {n}-gram model...")
        model = NGramModel(n)
        model.train(train_sentences)
        models[n] = model
        
        print(f"  Vocabulary size: {model.vocab_size}")
        print(f"  Unique {n}-grams: {len(model.ngram_counts)}")
        print(f"  Total {n}-gram tokens: {sum(model.ngram_counts.values())}")
    
    return models

def evaluate_interpolated_model(interpolation, test_sentences, dataset_name="Test"):
    """Evaluate interpolated model on a dataset"""
    print(f"\nEvaluating interpolated model on {dataset_name} set...")
    
    test_sample = test_sentences[:min(100, len(test_sentences))]
    interpolated_log_probs = []
    valid_sentences = 0
    
    for sentence in test_sample:
        tokens = sentence.strip().split()
        if len(tokens) < 4:
            continue
            
        prob = interpolation.get_sentence_probability(sentence)
        
        if prob > 0:
            log_prob = np.log(prob)
            interpolated_log_probs.append(log_prob)
            valid_sentences += 1
    
    if interpolated_log_probs:
        avg_log_prob = np.mean(interpolated_log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        print(f"Interpolated Model Performance on {dataset_name} Set:")
        print(f"  Valid sentences: {valid_sentences}/{len(test_sample)}")
        print(f"  Average Log Probability: {avg_log_prob:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        return {
            'avg_log_prob': avg_log_prob,
            'perplexity': perplexity,
            'valid_sentences': valid_sentences,
            'total_sentences': len(test_sample)
        }
    else:
        print(f"No valid sentences found for evaluation on {dataset_name} set.")
        return None

def compare_models(models, interpolation, test_sentences):
    """Compare individual models with interpolated model"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Evaluate individual models
    individual_results = evaluate_models_on_dataset(models, test_sentences, "Test")
    
    # Evaluate interpolated model
    interpolated_result = evaluate_interpolated_model(interpolation, test_sentences, "Test")
    
    print("\nComparison Summary:")
    print("-" * 30)
    for n in range(1, 5):
        if n in individual_results:
            result = individual_results[n]
            print(f"{n}-gram Model - Perplexity: {result['perplexity']:.2f}")
    
    if interpolated_result:
        print(f"Interpolated Model - Perplexity: {interpolated_result['perplexity']:.2f}")
        
        # Find best individual model
        best_individual = min(individual_results.items(), key=lambda x: x[1]['perplexity'])
        best_n, best_result = best_individual
        
        improvement = ((best_result['perplexity'] - interpolated_result['perplexity']) / 
                      best_result['perplexity']) * 100
        
        print(f"\nPerplexity Improvement:")
        print(f"  Best individual model: {best_n}-gram ({best_result['perplexity']:.2f})")
        print(f"  Interpolated model: {interpolated_result['perplexity']:.2f}")
        print(f"  Improvement: {improvement:.2f}%")

def analyze_lambda_weights(interpolation):
    """Analyze the learned interpolation weights"""
    print("\n" + "="*50)
    print("INTERPOLATION WEIGHTS ANALYSIS")
    print("="*50)
    
    print("Final optimized weights:")
    total_weight = sum(interpolation.lambdas)
    
    for i, weight in enumerate(interpolation.lambdas, 1):
        percentage = (weight / total_weight) * 100
        print(f"  λ{i} ({i}-gram model): {weight:.6f} ({percentage:.2f}%)")
    
    print(f"\nTotal weight sum: {total_weight:.6f}")
    
    # Interpret the weights
    print("\nInterpretation:")
    max_weight_idx = np.argmax(interpolation.lambdas)
    max_weight = interpolation.lambdas[max_weight_idx]
    
    print(f"- The {max_weight_idx + 1}-gram model has the highest weight ({max_weight:.4f})")
    
    if interpolation.lambdas[3] > 0.3:  # quadrigram weight
        print("- Strong reliance on quadrigram context suggests rich contextual patterns")
    elif interpolation.lambdas[0] > 0.3:  # unigram weight
        print("- High unigram weight suggests data sparsity issues")
    
    weight_entropy = -sum(w * np.log(w + 1e-10) for w in interpolation.lambdas if w > 0)
    print(f"- Weight entropy: {weight_entropy:.4f} (higher = more balanced usage)")

def main():
    """Main execution function for deleted interpolation"""
    print("=" * 80)
    print("LAB 5: DELETED INTERPOLATION SMOOTHING IMPLEMENTATION")
    print("=" * 80)
    
    # Step 1: Load sentences and create data splits
    print("\n1. LOADING DATA AND CREATING SPLITS")
    print("-" * 40)
    
    sentences = load_sentences()
    train_sentences, val_sentences, test_sentences = create_data_splits(
        sentences, val_size=1000, test_size=1000
    )
    
    print(f"\nData splits created:")
    print(f"  Training set: {len(train_sentences)} sentences")
    print(f"  Validation set: {len(val_sentences)} sentences")
    print(f"  Test set: {len(test_sentences)} sentences")
    
    if len(train_sentences) == 0:
        print("ERROR: No training data available. Please check your data files.")
        return
    
    # Step 2: Train individual n-gram models
    print("\n2. TRAINING INDIVIDUAL N-GRAM MODELS")
    print("-" * 45)
    
    models = train_individual_models(train_sentences)
    
    # Step 3: Initialize and optimize deleted interpolation
    print("\n3. DELETED INTERPOLATION OPTIMIZATION")
    print("-" * 45)
    
    # Use part of validation set for parameter optimization
    held_out = val_sentences[:min(500, len(val_sentences))]
    
    interpolation = DeletedInterpolation(
        models[1], models[2], models[3], models[4]
    )
    
    print(f"Using {len(held_out)} sentences for parameter optimization")
    interpolation.optimize_lambdas(held_out, iterations=10)
    
    # Step 4: Analyze interpolation weights
    analyze_lambda_weights(interpolation)
    
    # Step 5: Evaluate and compare models
    print("\n4. MODEL EVALUATION AND COMPARISON")
    print("-" * 40)
    
    compare_models(models, interpolation, test_sentences)
    
    # Step 6: Additional evaluation on validation set
    print("\n5. VALIDATION SET EVALUATION")
    print("-" * 35)
    
    val_result = evaluate_interpolated_model(interpolation, val_sentences, "Validation")
    
    # Step 7: Summary and insights
    print("\n6. SUMMARY AND KEY INSIGHTS")
    print("-" * 35)
    
    print("Deleted Interpolation Implementation Summary:")
    print("✓ Individual n-gram models trained with Good-Turing smoothing")
    print("✓ EM algorithm implemented for lambda optimization")
    print("✓ Interpolation weights optimized on held-out validation data")
    print("✓ Model comparison and evaluation completed")
    print("✓ Weight analysis and interpretation provided")
    
    print("\nKey Findings:")
    max_weight_model = np.argmax(interpolation.lambdas) + 1
    print(f"1. {max_weight_model}-gram model received highest interpolation weight")
    print(f"2. Deleted interpolation combines strengths of multiple n-gram orders")
    print(f"3. EM algorithm converges to optimal linear combination weights")
    print(f"4. Interpolation typically outperforms individual models")
    
    print("\n" + "="*80)
    print("DELETED INTERPOLATION SMOOTHING IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return interpolation, models

if __name__ == "__main__":
    main()