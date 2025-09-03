from collections import Counter

# smoothing.py

def add_one_smoothing(counts, vocab_size):
    """
    Applies Add-One (Laplace) smoothing to a dictionary of n-gram counts.
    Returns a dictionary of smoothed probabilities.
    """
    total = sum(counts.values()) + vocab_size
    all_ngrams = set(counts.keys())
    # Add unseen ngrams with count 0
    smoothed = {}
    for ng in all_ngrams:
        smoothed[ng] = (counts.get(ng, 0) + 1) / total
    return smoothed

def add_k_smoothing(counts, vocab_size, k=0.5):
    """
    Applies Add-K smoothing to a dictionary of n-gram counts.
    Returns a dictionary of smoothed probabilities.
    """
    total = sum(counts.values()) + k * vocab_size
    all_ngrams = set(counts.keys())
    smoothed = {}
    for ng in all_ngrams:
        smoothed[ng] = (counts.get(ng, 0) + k) / total
    return smoothed

def token_type_smoothing(counts, vocab_size=None):
    """
    Token Type Smoothing: Assigns probability based on n-gram type frequency.
    Not guaranteed to be a probability distribution.
    Returns a dictionary of smoothed values.
    """
    types = set(counts.keys())
    type_count = len(types)
    return {token: 1/type_count for token in counts}

def sentence_probability_smoothed(sentence_tokens, ngram_counts, vocab_size, smoothing_func, n, k=None):
    from LanguageModels import get_sentence_ngrams
    ngrams = get_sentence_ngrams(sentence_tokens, n)
    total_ngrams = sum(ngram_counts.values())
    if smoothing_func == add_k_smoothing and k is not None:
        smoothed_probs = smoothing_func(ngram_counts, vocab_size, k)
    else:
        smoothed_probs = smoothing_func(ngram_counts, vocab_size)
    prob = 1.0
    for ng in ngrams:
        prob *= smoothed_probs.get(ng, 1 / (total_ngrams + vocab_size))
    return prob

# Example usage:
if __name__ == "__main__":
    tokens = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
    counts = Counter(tokens)
    vocab_size = len(set(tokens))

    print("Add-One Smoothing:", add_one_smoothing(counts, vocab_size))
    print("Add-K Smoothing (k=0.5):", add_k_smoothing(counts, vocab_size, k=0.5))
    print("Token Type Smoothing:", token_type_smoothing(counts))