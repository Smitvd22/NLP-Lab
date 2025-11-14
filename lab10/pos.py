import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
import re
import math

class POSTagger:
    def __init__(self):
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.tag_set = set()
        self.vocab = set()
        self.start_tag = "<START>"
        self.end_tag = "<END>"
        
    def parse_sentence(self, sentence):
        """Parse a sentence in format 'word_tag word_tag ...' into (word, tag) pairs"""
        tokens = sentence.strip().split()
        words = []
        tags = []
        
        for token in tokens:
            # Handle special cases where underscore might be part of the word
            parts = token.rsplit('_', 1)
            if len(parts) == 2:
                word, tag = parts
                words.append(word.lower())  # Convert to lowercase for consistency
                tags.append(tag)
            else:
                # Skip malformed tokens
                continue
                
        return words, tags
    
    def train(self, sentences):
        """Train the HMM model on the given sentences"""
        # Reset counters
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.tag_set = set()
        self.vocab = set()
        
        # Count frequencies
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        
        for sentence in sentences:
            words, tags = self.parse_sentence(sentence)
            if not words or not tags:
                continue
                
            # Add start and end tags
            extended_tags = [self.start_tag] + tags + [self.end_tag]
            
            # Count transitions
            for i in range(len(extended_tags) - 1):
                curr_tag = extended_tags[i]
                next_tag = extended_tags[i + 1]
                transition_counts[curr_tag][next_tag] += 1
                self.tag_counts[curr_tag] += 1
                self.tag_set.add(curr_tag)
                self.tag_set.add(next_tag)
            
            # Count emissions
            for word, tag in zip(words, tags):
                emission_counts[tag][word] += 1
                self.word_counts[word] += 1
                self.vocab.add(word)
                self.tag_set.add(tag)
        
        # Calculate probabilities with add-one smoothing
        # Transition probabilities
        for tag in transition_counts:
            total_count = sum(transition_counts[tag].values())
            for next_tag in transition_counts[tag]:
                self.transition_probs[tag][next_tag] = (transition_counts[tag][next_tag] + 1) / (total_count + len(self.tag_set))
        
        # Add smoothing for unseen transitions
        for tag in self.tag_set:
            if tag not in transition_counts:
                continue
            total_count = sum(transition_counts[tag].values())
            for next_tag in self.tag_set:
                if next_tag not in self.transition_probs[tag]:
                    self.transition_probs[tag][next_tag] = 1 / (total_count + len(self.tag_set))
        
        # Emission probabilities
        for tag in emission_counts:
            total_count = sum(emission_counts[tag].values())
            for word in emission_counts[tag]:
                self.emission_probs[tag][word] = (emission_counts[tag][word] + 1) / (total_count + len(self.vocab) + 1)  # +1 for unknown words
            
            # Add probability for unknown words
            self.emission_probs[tag]["<UNK>"] = 1 / (total_count + len(self.vocab) + 1)
    
    def viterbi_decode(self, words):
        """Use Viterbi algorithm to find the most likely tag sequence"""
        if not words:
            return []
        
        # Convert words to lowercase
        words = [word.lower() for word in words]
        
        # Get all possible tags (excluding start/end tags for emission)
        possible_tags = [tag for tag in self.tag_set if tag not in [self.start_tag, self.end_tag]]
        
        if not possible_tags:
            return ['NN'] * len(words)  # Default to noun if no tags available
        
        # Initialize Viterbi table
        n = len(words)
        m = len(possible_tags)
        
        # dp[t][s] = probability of being in state s at time t
        dp = np.full((n, m), -np.inf)
        # backpointer[t][s] = previous state that led to state s at time t
        backpointer = np.zeros((n, m), dtype=int)
        
        # Initialize first column
        for i, tag in enumerate(possible_tags):
            word = words[0] if words[0] in self.vocab else "<UNK>"
            emission_prob = self.emission_probs[tag].get(word, self.emission_probs[tag].get("<UNK>", 1e-10))
            transition_prob = self.transition_probs[self.start_tag].get(tag, 1e-10)
            
            if emission_prob > 0 and transition_prob > 0:
                dp[0][i] = math.log(transition_prob) + math.log(emission_prob)
            else:
                dp[0][i] = -np.inf
        
        # Fill the rest of the table
        for t in range(1, n):
            for curr_state in range(m):
                curr_tag = possible_tags[curr_state]
                word = words[t] if words[t] in self.vocab else "<UNK>"
                emission_prob = self.emission_probs[curr_tag].get(word, self.emission_probs[curr_tag].get("<UNK>", 1e-10))
                
                if emission_prob == 0:
                    continue
                
                log_emission = math.log(emission_prob)
                
                for prev_state in range(m):
                    prev_tag = possible_tags[prev_state]
                    transition_prob = self.transition_probs[prev_tag].get(curr_tag, 1e-10)
                    
                    if transition_prob == 0 or dp[t-1][prev_state] == -np.inf:
                        continue
                    
                    log_transition = math.log(transition_prob)
                    prob = dp[t-1][prev_state] + log_transition + log_emission
                    
                    if prob > dp[t][curr_state]:
                        dp[t][curr_state] = prob
                        backpointer[t][curr_state] = prev_state
        
        # Find the best final state
        best_final_state = np.argmax(dp[n-1])
        
        # Backtrack to find the best path
        path = []
        state = best_final_state
        
        for t in range(n-1, -1, -1):
            path.append(possible_tags[state])
            if t > 0:
                state = backpointer[t][state]
        
        path.reverse()
        return path
    
    def predict(self, sentence):
        """Predict POS tags for a sentence"""
        # If sentence is already tokenized (list), use as is
        if isinstance(sentence, list):
            words = sentence
        else:
            # Simple tokenization for string input
            words = re.findall(r'\b\w+\b|[^\w\s]', sentence.lower())
        
        return self.viterbi_decode(words)

def load_data(filename):
    """Load the POS tagged data from file"""
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                sentences.append(line)
    return sentences

def k_fold_cross_validation(sentences, k=5):
    """Perform k-fold cross validation"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_true_tags = []
    all_pred_tags = []
    fold_results = []
    
    print(f"Performing {k}-fold cross validation...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(sentences)):
        print(f"\nFold {fold + 1}/{k}")
        
        # Split data
        train_sentences = [sentences[i] for i in train_idx]
        test_sentences = [sentences[i] for i in test_idx]
        
        # Train the model
        tagger = POSTagger()
        tagger.train(train_sentences)
        
        # Test the model
        fold_true_tags = []
        fold_pred_tags = []
        
        correct_predictions = 0
        total_predictions = 0
        
        for sentence in test_sentences:
            words, true_tags = tagger.parse_sentence(sentence)
            if not words or not true_tags:
                continue
                
            pred_tags = tagger.predict(words)
            
            # Ensure same length
            min_len = min(len(true_tags), len(pred_tags))
            true_tags = true_tags[:min_len]
            pred_tags = pred_tags[:min_len]
            
            fold_true_tags.extend(true_tags)
            fold_pred_tags.extend(pred_tags)
            
            # Calculate accuracy for this sentence
            for t, p in zip(true_tags, pred_tags):
                if t == p:
                    correct_predictions += 1
                total_predictions += 1
        
        # Calculate fold accuracy
        fold_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Store results
        all_true_tags.extend(fold_true_tags)
        all_pred_tags.extend(fold_pred_tags)
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'true_tags': fold_true_tags,
            'pred_tags': fold_pred_tags
        })
        
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
        print(f"Fold {fold + 1} Predictions: {total_predictions}")
    
    return fold_results, all_true_tags, all_pred_tags

def evaluate_performance(true_tags, pred_tags):
    """Calculate precision, recall, and F1-score"""
    # Get unique tags
    unique_tags = sorted(list(set(true_tags + pred_tags)))
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_tags, pred_tags, labels=unique_tags, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_tags, pred_tags, average='micro', zero_division=0
    )
    
    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(true_tags, pred_tags) if t == p) / len(true_tags)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'per_tag_precision': precision,
        'per_tag_recall': recall,
        'per_tag_f1': f1,
        'per_tag_support': support,
        'unique_tags': unique_tags
    }

def print_detailed_results(results, unique_tags):
    """Print detailed evaluation results"""
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-averaged Precision: {results['macro_precision']:.4f}")
    print(f"Macro-averaged Recall: {results['macro_recall']:.4f}")
    print(f"Macro-averaged F1-score: {results['macro_f1']:.4f}")
    print(f"Micro-averaged Precision: {results['micro_precision']:.4f}")
    print(f"Micro-averaged Recall: {results['micro_recall']:.4f}")
    print(f"Micro-averaged F1-score: {results['micro_f1']:.4f}")
    
    print(f"\nPer-tag Performance (Top 20 most frequent tags):")
    print(f"{'Tag':<8} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
    print("-" * 58)
    
    # Sort by support (frequency) and show top 20
    tag_metrics = list(zip(unique_tags, results['per_tag_precision'], 
                          results['per_tag_recall'], results['per_tag_f1'], 
                          results['per_tag_support']))
    tag_metrics.sort(key=lambda x: x[4], reverse=True)  # Sort by support
    
    for tag, precision, recall, f1, support in tag_metrics[:20]:
        print(f"{tag:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}")

def main():
    # Load data
    print("Loading POS tagged data...")
    filename = "wsj_pos_tagged_en.txt"
    sentences = load_data(filename)
    print(f"Loaded {len(sentences)} sentences")
    
    # Perform k-fold cross validation
    k = 5  # You can change this to any value >= 3
    fold_results, all_true_tags, all_pred_tags = k_fold_cross_validation(sentences, k)
    
    # Calculate overall performance metrics
    print("\nCalculating overall performance metrics...")
    results = evaluate_performance(all_true_tags, all_pred_tags)
    
    # Print results
    print_detailed_results(results, results['unique_tags'])
    
    # Print fold-wise results
    print(f"\n{'='*80}")
    print("FOLD-WISE RESULTS")
    print("="*80)
    
    fold_accuracies = []
    for fold_result in fold_results:
        fold_metrics = evaluate_performance(fold_result['true_tags'], fold_result['pred_tags'])
        fold_accuracies.append(fold_metrics['accuracy'])
        
        print(f"\nFold {fold_result['fold']}:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {fold_metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {fold_metrics['micro_f1']:.4f}")
    
    print(f"\nCross-validation Summary:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    
    # Demonstrate the tagger on a sample sentence
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    # Train a final model on all data for demonstration
    tagger = POSTagger()
    tagger.train(sentences)
    
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog .",
        "I love natural language processing .",
        "Stock prices rose yesterday ."
    ]
    
    for sentence in sample_sentences:
        words = sentence.lower().split()
        pred_tags = tagger.predict(words)
        
        print(f"\nSentence: {sentence}")
        print("Predicted tags:")
        for word, tag in zip(words, pred_tags):
            print(f"  {word} -> {tag}")

if __name__ == "__main__":
    main()
