"""
NAIVE BAYES CLASSIFIER - DETAILED STEP-BY-STEP ANALYSIS
========================================================

This file provides a complete breakdown of how the Naive Bayes classifier works
with detailed calculations for each step.
"""

import numpy as np
from collections import Counter

# Training Data
print("=" * 80)
print("STEP 1: TRAINING DATA")
print("=" * 80)

X_train = [
    "offer money prince nigeria",      # spam
    "nigeria prince money back",       # spam  
    "claim your free money now",       # spam
    "meeting report status update",    # not_spam
    "project status meeting today",    # not_spam
    "please review this report"        # not_spam
]

y_train = ["spam", "spam", "spam", "not_spam", "not_spam", "not_spam"]

print("Training Documents:")
for i, (text, label) in enumerate(zip(X_train, y_train)):
    print(f"  Document {i+1}: '{text}' -> {label}")

total_docs = len(y_train)
print(f"\nTotal Documents: {total_docs}")

# Step 2: Calculate Prior Probabilities
print("\n" + "=" * 80)
print("STEP 2: CALCULATE PRIOR PROBABILITIES")
print("=" * 80)

class_counts = Counter(y_train)
print(f"Class Counts: {dict(class_counts)}")

priors = {}
for cls, count in class_counts.items():
    priors[cls] = count / total_docs
    print(f"P({cls}) = {count}/{total_docs} = {priors[cls]:.4f}")

# Step 3: Build Vocabulary and Word Counts
print("\n" + "=" * 80)
print("STEP 3: BUILD VOCABULARY AND WORD COUNTS")
print("=" * 80)

vocabulary = set()
word_counts_by_class = {"spam": {}, "not_spam": {}}

print("Processing each document:")
for i, (text, cls) in enumerate(zip(X_train, y_train)):
    words = text.lower().split()
    print(f"  Document {i+1} ({cls}): {words}")
    
    for word in words:
        vocabulary.add(word)
        word_counts_by_class[cls][word] = word_counts_by_class[cls].get(word, 0) + 1

print(f"\nVocabulary (sorted): {sorted(vocabulary)}")
print(f"Vocabulary size: {len(vocabulary)}")

print(f"\nWord counts by class:")
for cls in ["spam", "not_spam"]:
    print(f"  {cls}: {dict(sorted(word_counts_by_class[cls].items()))}")
    total_words = sum(word_counts_by_class[cls].values())
    print(f"    Total words in {cls}: {total_words}")

# Step 4: Calculate Likelihoods with Laplace Smoothing
print("\n" + "=" * 80)
print("STEP 4: CALCULATE LIKELIHOODS (with Laplace Smoothing α=1)")
print("=" * 80)

alpha = 1
vocab_size = len(vocabulary)
likelihoods = {"spam": {}, "not_spam": {}}

print("Formula: P(word|class) = (count(word, class) + α) / (total_words_in_class + α × vocab_size)")
print(f"α = {alpha}, vocab_size = {vocab_size}")

for cls in ["spam", "not_spam"]:
    total_words_in_class = sum(word_counts_by_class[cls].values())
    print(f"\nClass: {cls}")
    print(f"Total words in class: {total_words_in_class}")
    print(f"Denominator = {total_words_in_class} + {alpha} × {vocab_size} = {total_words_in_class + alpha * vocab_size}")
    
    print("Word likelihoods:")
    for word in sorted(vocabulary):
        word_count = word_counts_by_class[cls].get(word, 0)
        numerator = word_count + alpha
        denominator = total_words_in_class + (alpha * vocab_size)
        likelihood = numerator / denominator
        likelihoods[cls][word] = likelihood
        print(f"  P('{word}'|{cls}) = ({word_count} + {alpha}) / {denominator} = {numerator}/{denominator} = {likelihood:.6f}")

# Step 5: Make Predictions
print("\n" + "=" * 80)
print("STEP 5: MAKING PREDICTIONS")
print("=" * 80)

X_test = [
    "nigeria money back guarantee",
    "please send the meeting report", 
    "free offer for you"
]

print("Test Documents:")
for i, text in enumerate(X_test):
    print(f"  Test {i+1}: '{text}'")

print("\nPrediction Process:")
print("Formula: P(class|document) ∝ P(class) × ∏P(word|class)")
print("Using log probabilities to avoid underflow:")
print("log P(class|document) = log P(class) + Σ log P(word|class)")

for test_idx, text in enumerate(X_test):
    print(f"\n--- TEST DOCUMENT {test_idx + 1}: '{text}' ---")
    words = text.lower().split()
    print(f"Words: {words}")
    
    posteriors = {}
    
    for cls in ["spam", "not_spam"]:
        print(f"\nCalculating for class '{cls}':")
        
        # Start with log prior
        log_posterior = np.log(priors[cls])
        print(f"  log P({cls}) = log({priors[cls]:.4f}) = {log_posterior:.6f}")
        
        # Add log likelihood for each word
        word_contributions = []
        for word in words:
            if word in vocabulary:
                log_likelihood = np.log(likelihoods[cls][word])
                log_posterior += log_likelihood
                word_contributions.append(f"log P('{word}'|{cls}) = {log_likelihood:.6f}")
            else:
                print(f"    Word '{word}' not in vocabulary - skipping")
        
        print(f"  Word contributions:")
        for contrib in word_contributions:
            print(f"    {contrib}")
        
        posteriors[cls] = log_posterior
        print(f"  Total log P({cls}|document) = {log_posterior:.6f}")
    
    # Find the class with highest posterior
    predicted_class = max(posteriors, key=posteriors.get)
    print(f"\nPredicted class: {predicted_class} (highest log posterior)")
    print(f"Posteriors: {', '.join([f'{cls}: {post:.6f}' for cls, post in posteriors.items()])}")

# Summary of the Algorithm
print("\n" + "=" * 80)
print("ALGORITHM SUMMARY")
print("=" * 80)
print("""
Naive Bayes Algorithm Steps:

1. TRAINING PHASE:
   a) Calculate prior probabilities: P(class) = count(class) / total_documents
   b) Build vocabulary from all training documents
   c) Count word occurrences for each class
   d) Calculate likelihoods with Laplace smoothing:
      P(word|class) = (count(word, class) + α) / (total_words_in_class + α × vocab_size)

2. PREDICTION PHASE:
   a) For each test document, calculate log posterior for each class:
      log P(class|document) = log P(class) + Σ log P(word|class)
   b) Choose the class with the highest log posterior probability

Key Assumptions:
- Words are independent given the class (Naive assumption)
- Laplace smoothing handles unseen words
- Log probabilities prevent numerical underflow

Alpha (α) parameter:
- α = 1: Standard Laplace smoothing
- α < 1: Less smoothing (more aggressive)  
- α > 1: More smoothing (more conservative)
""")

print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)