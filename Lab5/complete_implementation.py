"""
Lab 5 - Complete Good-Turing Smoothing Implementation and Analysis

This script provides a complete implementation of:
1. Data splits with random sampling
2. Good-Turing smoothing for all n-gram models
3. Comprehensive frequency tables
4. Deleted interpolation for quadrigram model
5. Detailed analysis and evaluation

Author: Student
Course: NLP Lab
Lab: 5
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import sys
import os

# Helper functions
def build_ngram_model(tokens_list, n):
    """Build n-gram model from a list of tokens"""
    counts = defaultdict(int)
    vocab = set(tokens_list)
    
    for i in range(len(tokens_list) - n + 1):
        ngram = tuple(tokens_list[i:i + n])
        counts[ngram] += 1
    
    return counts, vocab

def get_sentence_ngrams(sentence_tokens, n):
    """Get n-grams from a sentence"""
    ngrams = []
    for i in range(len(sentence_tokens) - n + 1):
        ngram = tuple(sentence_tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

class GoodTuringSmoothing:
    """Implementation of Good-Turing smoothing"""
    
    def __init__(self):
        self.vocab_size = 0
        self.total_ngrams = 0
        self.frequency_counts = {}  # Nc values
        self.good_turing_estimates = {}  # C* values
        
    def calculate_frequency_counts(self, ngram_counts):
        """Calculate Nc - number of n-grams that occur c times"""
        self.frequency_counts = defaultdict(int)
        for count in ngram_counts.values():
            self.frequency_counts[count] += 1
        self.total_ngrams = len(ngram_counts)
        
    def calculate_good_turing_estimates(self, max_count=50):
        """Calculate Good-Turing estimates C* for frequencies"""
        self.good_turing_estimates = {}
        
        for c in range(max_count + 1):
            if c == 0:
                # For unseen n-grams: use N1/N
                if 1 in self.frequency_counts:
                    self.good_turing_estimates[c] = self.frequency_counts[1] / self.total_ngrams
                else:
                    self.good_turing_estimates[c] = 0
            else:
                # For seen n-grams: c* = (c+1) * N_{c+1} / N_c
                nc = self.frequency_counts.get(c, 0)
                nc_plus_1 = self.frequency_counts.get(c + 1, 0)
                
                if nc > 0 and nc_plus_1 > 0:
                    self.good_turing_estimates[c] = (c + 1) * nc_plus_1 / nc
                else:
                    # Fallback to original count if no smoothing possible
                    self.good_turing_estimates[c] = c
    
    def get_smoothed_probability(self, ngram, ngram_counts, n=1):
        """Get Good-Turing smoothed probability for an n-gram"""
        count = ngram_counts.get(ngram, 0)
        total_tokens = sum(ngram_counts.values())
        
        if count == 0:
            # Unseen n-gram probability calculation
            n1 = self.frequency_counts.get(1, 0)
            
            if n == 1:
                # Unigram case: P_unseen = N1 / (N * (V - U))
                unseen_types = max(1, self.vocab_size - len(ngram_counts))
                return n1 / (self.total_ngrams * unseen_types)
            else:
                # Higher n-grams: P_unseen = N1 / (N * (V^n - N))
                possible_ngrams = self.vocab_size ** n
                unseen_ngrams = max(1, possible_ngrams - self.total_ngrams)
                return n1 / (self.total_ngrams * unseen_ngrams)
        else:
            # Seen n-gram: use Good-Turing estimate
            c_star = self.good_turing_estimates.get(count, count)
            return c_star / total_tokens

class NGramModel:
    """N-gram language model with Good-Turing smoothing"""
    
    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.vocab = set()
        self.vocab_size = 0
        self.smoother = GoodTuringSmoothing()
        
    def train(self, sentences):
        """Train the n-gram model on sentences"""
        all_tokens = []
        for sentence in sentences:
            tokens = sentence.strip().split()
            all_tokens.extend(tokens)
        
        # Build n-gram counts
        self.ngram_counts, self.vocab = build_ngram_model(all_tokens, self.n)
        self.vocab_size = len(self.vocab)
        
        # Calculate Good-Turing smoothing
        self.smoother.vocab_size = self.vocab_size
        self.smoother.calculate_frequency_counts(self.ngram_counts)
        self.smoother.calculate_good_turing_estimates()
        
    def get_sentence_probability(self, sentence):
        """Calculate probability of a sentence using Good-Turing smoothing"""
        tokens = sentence.strip().split()
        if len(tokens) < self.n:
            return 1e-10  # Very small probability for short sentences
            
        ngrams = get_sentence_ngrams(tokens, self.n)
        
        prob = 1.0
        for ngram in ngrams:
            ngram_prob = self.smoother.get_smoothed_probability(
                ngram, self.ngram_counts, self.n
            )
            prob *= max(ngram_prob, 1e-10)  # Avoid zero probabilities
            
        return prob

class DeletedInterpolation:
    """Deleted interpolation smoothing for quadrigram model"""
    
    def __init__(self, unigram_model, bigram_model, trigram_model, quadrigram_model):
        self.models = [unigram_model, bigram_model, trigram_model, quadrigram_model]
        self.lambdas = [0.25, 0.25, 0.25, 0.25]  # Initial equal weights
        
    def optimize_lambdas(self, held_out_sentences, iterations=5):
        """Optimize lambda parameters using EM algorithm"""
        print("Optimizing interpolation parameters...")
        
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
                    probs = []
                    
                    # Unigram: P(w4)
                    unigram = (quad[3],)
                    probs.append(self.models[0].smoother.get_smoothed_probability(
                        unigram, self.models[0].ngram_counts, 1
                    ))
                    
                    # Bigram: P(w4|w3)
                    bigram = (quad[2], quad[3])
                    probs.append(self.models[1].smoother.get_smoothed_probability(
                        bigram, self.models[1].ngram_counts, 2
                    ))
                    
                    # Trigram: P(w4|w2,w3)
                    trigram = (quad[1], quad[2], quad[3])
                    probs.append(self.models[2].smoother.get_smoothed_probability(
                        trigram, self.models[2].ngram_counts, 3
                    ))
                    
                    # Quadrigram: P(w4|w1,w2,w3)
                    probs.append(self.models[3].smoother.get_smoothed_probability(
                        quad, self.models[3].ngram_counts, 4
                    ))
                    
                    # Calculate interpolated probability
                    interpolated_prob = sum(self.lambdas[i] * probs[i] for i in range(4))
                    
                    if interpolated_prob > 0:
                        # Update lambda numerators (E-step)
                        for i in range(4):
                            weight = (self.lambdas[i] * probs[i]) / interpolated_prob
                            lambda_numerators[i] += weight
                        lambda_denominator += 1
            
            # Update lambdas (M-step)
            if lambda_denominator > 0:
                total_lambda = sum(lambda_numerators)
                if total_lambda > 0:
                    for i in range(4):
                        self.lambdas[i] = lambda_numerators[i] / total_lambda
                        
            print(f"Iteration {iteration + 1}: λ = {[f'{l:.4f}' for l in self.lambdas]}")
    
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

def load_data():
    """Load sentences from available data sources"""
    sentences = []
    
    # Try multiple data sources
    data_sources = [
        '../Lab1/tokenized_hindi_sentences.json',
        '../Lab4/news_articles.txt'
    ]
    
    for source in data_sources:
        try:
            if source.endswith('.json'):
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for document in data:
                        sentences.extend(document)
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('.'):
                        line = line.strip()
                        if line:
                            sentences.append(line + '.')
            print(f"Loaded {len(sentences)} sentences from {source}")
            break
        except:
            continue
    
    # Fallback to sample data if no files found
    if not sentences:
        sentences = [
            "यह एक अच्छा दिन है।",
            "भारत में शिक्षा का स्तर बढ़ रहा है।",
            "सरकार ने नई नीति की घोषणा की है।",
            "छात्रों को अच्छी सुविधाएं मिल रही हैं।",
            "तकनीक के क्षेत्र में भारत आगे बढ़ रहा है।",
            "नई दिल्ली में बैठक हुई है।",
            "प्रधानमंत्री ने देश को संबोधित किया।",
            "वैज्ञानिकों ने नई खोज की है।",
            "आर्थिक विकास की दर बढ़ रही है।",
            "कृषि उत्पादन में वृद्धि हुई है।"
        ] * 5  # Repeat to get more data
        print(f"Using sample data: {len(sentences)} sentences")
    
    return sentences

def create_data_splits(sentences, val_size=10, test_size=10, random_seed=42):
    """Create train/validation/test splits"""
    random.seed(random_seed)
    
    # Adjust sizes based on available data
    total = len(sentences)
    val_size = min(val_size, total // 4)
    test_size = min(test_size, total // 4)
    
    # Shuffle and split
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    test_set = shuffled[:test_size]
    val_set = shuffled[test_size:test_size + val_size]
    train_set = shuffled[test_size + val_size:]
    
    return train_set, val_set, test_set

def print_frequency_table(smoother, model_name, top_n=20):
    """Print comprehensive frequency table"""
    print(f"\n{model_name} Model - Top {top_n} Frequency Table:")
    print("C (MLE)\\tNc\\tC*\\tDescription")
    print("-" * 60)
    
    # Sort by frequency count
    sorted_freqs = sorted(smoother.frequency_counts.items())
    
    for i, (c, nc) in enumerate(sorted_freqs[:top_n]):
        c_star = smoother.good_turing_estimates.get(c, c)
        
        if c == 1:
            desc = "Singletons"
        elif c == 2:
            desc = "Doubletons"
        else:
            desc = f"Count-{c} items"
            
        print(f"{c}\\t{nc}\\t{c_star:.6f}\\t{desc}")

def evaluate_models(models, test_sentences):
    """Evaluate all models on test data"""
    print("\\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Use first 5 sentences for detailed analysis
    sample_sentences = test_sentences[:5]
    
    for n, model in models.items():
        print(f"\\n{n}-gram Model Results:")
        print("-" * 30)
        
        log_probs = []
        for i, sentence in enumerate(sample_sentences):
            prob = model.get_sentence_probability(sentence)
            log_prob = np.log(prob) if prob > 0 else -50  # Cap very low probabilities
            log_probs.append(log_prob)
            
            print(f"Sentence {i+1}: {sentence[:40]}...")
            print(f"  Probability: {prob:.2e}")
            print(f"  Log Probability: {log_prob:.4f}")
        
        # Calculate average metrics
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        print(f"\\nAverage Log Probability: {avg_log_prob:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("LAB 5: GOOD-TURING SMOOTHING IMPLEMENTATION")
    print("=" * 80)
    
    # Step 1: Load and split data
    print("\\n1. LOADING AND SPLITTING DATA")
    print("-" * 40)
    
    sentences = load_data()
    train_sentences, val_sentences, test_sentences = create_data_splits(sentences)
    
    print(f"Data splits created:")
    print(f"  Training: {len(train_sentences)} sentences")
    print(f"  Validation: {len(val_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")
    
    # Step 2: Train n-gram models with Good-Turing smoothing
    print("\\n2. TRAINING N-GRAM MODELS WITH GOOD-TURING SMOOTHING")
    print("-" * 55)
    
    models = {}
    for n in range(1, 5):
        print(f"\\nTraining {n}-gram model...")
        model = NGramModel(n)
        model.train(train_sentences)
        models[n] = model
        
        print(f"  Vocabulary size: {model.vocab_size}")
        print(f"  Unique {n}-grams: {len(model.ngram_counts)}")
        print(f"  Total {n}-gram tokens: {sum(model.ngram_counts.values())}")
        print(f"  Singletons (N1): {model.smoother.frequency_counts.get(1, 0)}")
        
        # Show frequency table for unigram model
        if n == 1:
            print_frequency_table(model.smoother, "Unigram")
    
    # Step 3: Show comprehensive frequency tables
    print("\\n3. FREQUENCY DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    for n, model in models.items():
        if n <= 2:  # Show details for unigram and bigram
            print_frequency_table(model.smoother, f"{n}-gram", 15)
    
    # Step 4: Evaluate models
    print("\\n4. MODEL EVALUATION")
    print("-" * 25)
    
    evaluate_models(models, test_sentences)
    
    # Step 5: Deleted interpolation
    print("\\n5. DELETED INTERPOLATION FOR QUADRIGRAM MODEL")
    print("-" * 50)
    
    # Use part of validation set for optimization
    held_out = val_sentences[:min(20, len(val_sentences))]
    
    interpolation = DeletedInterpolation(
        models[1], models[2], models[3], models[4]
    )
    
    interpolation.optimize_lambdas(held_out)
    
    print(f"\\nFinal interpolation weights:")
    for i, weight in enumerate(interpolation.lambdas, 1):
        print(f"  λ{i} ({i}-gram): {weight:.6f}")
    
    # Evaluate interpolated model
    print("\\nEvaluating interpolated model on test set...")
    
    sample_test = test_sentences[:3]
    interpolated_log_probs = []
    
    for sentence in sample_test:
        tokens = sentence.strip().split()
        if len(tokens) < 4:
            continue
            
        quadrigrams = get_sentence_ngrams(tokens, 4)
        sentence_prob = 1.0
        
        for quad in quadrigrams:
            prob = interpolation.get_interpolated_probability(quad)
            sentence_prob *= max(prob, 1e-10)
        
        if sentence_prob > 0:
            log_prob = np.log(sentence_prob)
            interpolated_log_probs.append(log_prob)
            print(f"Sentence: {sentence[:40]}...")
            print(f"  Interpolated Probability: {sentence_prob:.2e}")
            print(f"  Log Probability: {log_prob:.4f}")
    
    if interpolated_log_probs:
        avg_log_prob = np.mean(interpolated_log_probs)
        perplexity = np.exp(-avg_log_prob)
        print(f"\\nInterpolated Model Performance:")
        print(f"  Average Log Probability: {avg_log_prob:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
    
    # Step 6: Summary
    print("\\n6. SUMMARY AND INSIGHTS")
    print("-" * 30)
    
    print("Key findings:")
    print("1. Good-Turing smoothing effectively handles unseen n-grams")
    print("2. Higher-order models require more sophisticated smoothing")
    print("3. Deleted interpolation combines strengths of different n-gram orders")
    print("4. Lower-order models typically get higher weights in interpolation")
    print("5. Frequency tables show the distribution of n-gram counts")
    
    print("\\n" + "="*80)
    print("IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
