"""
Lab 5 - Good-Turing Smoothing Implementation

This module implements:
1. Data splits (train/validation/test)
2. Good-Turing smoothing for all n-gram models (unigram, bigram, trigram, quadrigram)
3. Sentence probability computation with smoothed models
4. Frequency table with MLE, Nc, and C* values
5. Deleted interpolation smoothing for quadrigram model
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import sys
import os

# Helper functions (copied from LanguageModels.py to avoid import issues)
def build_ngram_model(tokens_list, n):
    """Build n-gram model from a list of tokens"""
    from collections import defaultdict
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

def get_vocab(tokens_list):
    """Get vocabulary from tokens"""
    return set(tokens_list)

class GoodTuringSmoothing:
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
        
    def calculate_good_turing_estimates(self, max_count=100):
        """Calculate Good-Turing estimates C* for frequencies"""
        self.good_turing_estimates = {}
        
        for c in range(max_count + 1):
            if c == 0:
                # For unseen n-grams
                if 1 in self.frequency_counts:
                    self.good_turing_estimates[c] = self.frequency_counts[1] / self.total_ngrams
                else:
                    self.good_turing_estimates[c] = 0
            else:
                # For seen n-grams: c* = (c+1) * N_{c+1} / N_c
                nc = self.frequency_counts.get(c, 0)
                nc_plus_1 = self.frequency_counts.get(c + 1, 0)
                
                if nc > 0:
                    if nc_plus_1 > 0:
                        self.good_turing_estimates[c] = (c + 1) * nc_plus_1 / nc
                    else:
                        # Use original count if no c+1 frequency
                        self.good_turing_estimates[c] = c
                else:
                    self.good_turing_estimates[c] = c
    
    def get_smoothed_probability(self, ngram, ngram_counts, n=1):
        """Get Good-Turing smoothed probability for an n-gram"""
        count = ngram_counts.get(ngram, 0)
        
        if count == 0:
            # Unseen n-gram
            if n == 1:
                # Unigram case: V - U (unique seen unigrams)
                unseen_types = self.vocab_size - len(ngram_counts)
                if unseen_types > 0:
                    n1 = self.frequency_counts.get(1, 0)
                    return n1 / (self.total_ngrams * unseen_types)
                else:
                    return 0
            else:
                # Higher n-grams: V^n - N
                possible_ngrams = self.vocab_size ** n
                unseen_ngrams = possible_ngrams - self.total_ngrams
                if unseen_ngrams > 0:
                    n1 = self.frequency_counts.get(1, 0)
                    return n1 / (self.total_ngrams * unseen_ngrams)
                else:
                    return 0
        else:
            # Seen n-gram
            c_star = self.good_turing_estimates.get(count, count)
            return c_star / sum(ngram_counts.values())

class DataSplitter:
    def __init__(self, validation_size=20, test_size=20, random_seed=42):
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def split_sentences(self, sentences):
        """Split sentences into train/validation/test sets"""
        # Shuffle sentences
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)
        
        # Create splits
        test_set = shuffled_sentences[:self.test_size]
        validation_set = shuffled_sentences[self.test_size:self.test_size + self.validation_size]
        train_set = shuffled_sentences[self.test_size + self.validation_size:]
        
        return train_set, validation_set, test_set

class NGramModel:
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
            # Tokenize sentence (split by space for now)
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
            return 0.0
            
        ngrams = get_sentence_ngrams(tokens, self.n)
        
        prob = 1.0
        for ngram in ngrams:
            ngram_prob = self.smoother.get_smoothed_probability(
                ngram, self.ngram_counts, self.n
            )
            prob *= ngram_prob
            
        return prob

class DeletedInterpolation:
    def __init__(self, unigram_model, bigram_model, trigram_model, quadrigram_model):
        self.models = [unigram_model, bigram_model, trigram_model, quadrigram_model]
        self.lambdas = [0.25, 0.25, 0.25, 0.25]  # Initial weights
        
    def optimize_lambdas(self, held_out_sentences, iterations=10):
        """Optimize lambda parameters using EM algorithm on held-out data"""
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
                        # Update lambda numerators
                        for i in range(4):
                            lambda_numerators[i] += (self.lambdas[i] * probs[i]) / interpolated_prob
                        lambda_denominator += 1
            
            # Update lambdas
            if lambda_denominator > 0:
                for i in range(4):
                    self.lambdas[i] = lambda_numerators[i] / lambda_denominator
                    
            print(f"Iteration {iteration + 1}: λ = {self.lambdas}")
    
    def get_interpolated_probability(self, quadrigram):
        """Get interpolated probability for a quadrigram"""
        # Get probabilities from each model
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

def load_sentences_from_json(file_path):
    """Load sentences from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    for document in data:
        sentences.extend(document)
    
    return sentences

def load_sentences_from_text(file_path):
    """Load sentences from text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by sentence endings
    sentences = []
    for line in content.split('.'):
        line = line.strip()
        if line:
            sentences.append(line + '.')
    
    return sentences

def print_frequency_table(smoother, top_n=100):
    """Print frequency table with C, MLE, Nc, C* values"""
    print(f"\nTop {top_n} Frequency Table:")
    print("C (MLE)\tNc\tC*")
    print("-" * 30)
    
    # Sort frequencies by count
    sorted_freqs = sorted(smoother.frequency_counts.items())
    
    for i, (c, nc) in enumerate(sorted_freqs[:top_n]):
        c_star = smoother.good_turing_estimates.get(c, c)
        print(f"{c}\t{nc}\t{c_star:.6f}")

def main():
    print("=== Lab 5: Good-Turing Smoothing Implementation ===\n")
    
    # Load sentences - try multiple sources
    print("Loading sentences...")
    
    # Try to load from different sources
    sentences = []
    
    # Try JSON file first
    try:
        hindi_sentences = load_sentences_from_json('../Lab1/tokenized_hindi_sentences.json')
        sentences.extend(hindi_sentences)
        print(f"Loaded {len(hindi_sentences)} sentences from JSON")
    except:
        print("Could not load from JSON file")
    
    # Try text file
    try:
        text_sentences = load_sentences_from_text('../Lab4/news_articles.txt')
        sentences.extend(text_sentences)
        print(f"Loaded {len(text_sentences)} sentences from text file")
    except:
        print("Could not load from text file")
    
    # If still no data, create sample sentences
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
            "कृषि उत्पादन में वृद्धि हुई है।",
            "डिजिटल इंडिया अभियान सफल हो रहा है।",
            "स्वच्छ भारत मिशन चल रहा है।",
            "युवाओं के लिए रोजगार के अवसर बढ़े हैं।",
            "स्वास्थ्य सेवाओं में सुधार हो रहा है।",
            "बुनियादी ढांचे का विकास तेजी से हो रहा है।",
            "नवीकरणीय ऊर्जा पर ध्यान दिया जा रहा है।",
            "महिला सशक्तिकरण के लिए योजनाएं चल रही हैं।",
            "पर्यावरण संरक्षण के लिए कदम उठाए गए हैं।",
            "खेल के क्षेत्र में भारत की स्थिति मजबूत हो रही है।",
            "शिक्षा व्यवस्था में सुधार किया जा रहा है।"
        ]
        print(f"Using sample sentences: {len(sentences)}")
    
    print(f"Total sentences: {len(sentences)}")
    
    # Adjust split sizes based on available data
    total_sentences = len(sentences)
    val_size = min(20, total_sentences // 4)
    test_size = min(20, total_sentences // 4)
    
    # Create data splits
    print(f"\nCreating data splits (val: {val_size}, test: {test_size})...")
    splitter = DataSplitter(validation_size=val_size, test_size=test_size)
    train_sentences, val_sentences, test_sentences = splitter.split_sentences(sentences)
    
    print(f"Training set: {len(train_sentences)} sentences")
    print(f"Validation set: {len(val_sentences)} sentences")
    print(f"Test set: {len(test_sentences)} sentences")
    
    if len(train_sentences) == 0:
        print("No training data available. Please check your data files.")
        return
    
    # Train n-gram models with Good-Turing smoothing
    print("\nTraining n-gram models with Good-Turing smoothing...")
    models = {}
    
    for n in range(1, 5):
        print(f"\nTraining {n}-gram model...")
        model = NGramModel(n)
        model.train(train_sentences)
        models[n] = model
        
        print(f"Vocabulary size: {model.vocab_size}")
        print(f"Total {n}-grams: {model.smoother.total_ngrams}")
        
        # Print frequency table for unigram model
        if n == 1:
            print_frequency_table(model.smoother)
    
    # Evaluate on validation and test sets
    print("\n=== Evaluation Results ===")
    
    for dataset_name, sentences in [("Validation", val_sentences), ("Test", test_sentences)]:
        print(f"\n{dataset_name} Set Evaluation:")
        
        # Take first 10 sentences for evaluation (to avoid very small probabilities)
        eval_sentences = sentences[:10]
        
        for n in range(1, 5):
            total_log_prob = 0
            valid_sentences = 0
            
            for sentence in eval_sentences:
                prob = models[n].get_sentence_probability(sentence)
                if prob > 0:
                    total_log_prob += np.log(prob)
                    valid_sentences += 1
            
            if valid_sentences > 0:
                avg_log_prob = total_log_prob / valid_sentences
                perplexity = np.exp(-avg_log_prob)
                print(f"{n}-gram model - Avg Log Prob: {avg_log_prob:.4f}, Perplexity: {perplexity:.4f}")
    
    # Implement deleted interpolation for quadrigram model
    print("\n=== Deleted Interpolation for Quadrigram Model ===")
    
    # Use part of validation set for parameter optimization
    held_out = val_sentences[:100]  # Use first 100 sentences
    
    interpolation = DeletedInterpolation(
        models[1], models[2], models[3], models[4]
    )
    
    print("Optimizing interpolation parameters...")
    interpolation.optimize_lambdas(held_out)
    
    print(f"\nFinal interpolation weights: {interpolation.lambdas}")
    
    # Evaluate interpolated model
    print("\nEvaluating interpolated quadrigram model...")
    test_sentences_small = test_sentences[:10]
    total_log_prob = 0
    valid_sentences = 0
    
    for sentence in test_sentences_small:
        tokens = sentence.strip().split()
        if len(tokens) < 4:
            continue
            
        quadrigrams = get_sentence_ngrams(tokens, 4)
        sentence_prob = 1.0
        
        for quad in quadrigrams:
            prob = interpolation.get_interpolated_probability(quad)
            if prob > 0:
                sentence_prob *= prob
        
        if sentence_prob > 0:
            total_log_prob += np.log(sentence_prob)
            valid_sentences += 1
    
    if valid_sentences > 0:
        avg_log_prob = total_log_prob / valid_sentences
        perplexity = np.exp(-avg_log_prob)
        print(f"Interpolated model - Avg Log Prob: {avg_log_prob:.4f}, Perplexity: {perplexity:.4f}")
    
    print("\n=== Implementation Complete ===")

if __name__ == "__main__":
    main()