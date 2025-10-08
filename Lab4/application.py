import os
import random
import json
import math
from LanguageModels import build_ngram_model, get_vocab, get_ngram_vocab, get_sentence_ngrams

# Path to a folder containing news articles in your language (as plain text files)
NEWS_ARTICLE = "lab4/news_articles.txt"
SMOOTHED_RESULTS_FILE = "lab4/smoothing_results/hindi_smoothing_complete_results.json"

def get_sentences_from_articles(file_path, num_sentences=1000):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        file_sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences.extend(file_sentences)
    random.shuffle(sentences)
    return sentences[:num_sentences]

def load_smoothed_results():
    """Load pre-computed smoothed probabilities from JSON file"""
    with open(SMOOTHED_RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_ngrams(tokens, n):
    """Generate n-grams from a list of tokens"""
    if n == 1:
        return tokens
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calculate_sentence_probability_from_smoothed(tokens, n, smoothing_method, smoothed_data):
    """Calculate sentence probability using pre-computed smoothed probabilities"""
    if n == 1:
        # For unigrams, just use the tokens directly
        ngrams = tokens
    else:
        # For n-grams > 1, create n-grams from the tokens
        ngrams = get_ngrams(tokens, n)
    
    if not ngrams:
        return 0.0
    
    # Get the smoothed probabilities for this n-gram order
    ngram_data = smoothed_data[f"{n}_gram"]
    smoothed_probs = ngram_data["smoothed_probabilities"][smoothing_method]
    
    # Calculate log probability to avoid underflow
    log_prob = 0.0
    for ngram in ngrams:
        if ngram in smoothed_probs:
            prob = smoothed_probs[ngram]
            if prob > 0:
                log_prob += math.log(prob)
            else:
                # Use a very small probability for zero probabilities
                log_prob += math.log(1e-10)
        else:
            # For unseen n-grams, use a very small probability
            log_prob += math.log(1e-10)
    
    # Convert back to probability (will be very small for long sentences)
    return math.exp(log_prob)

def main():
    print("=== SENTENCE PROBABILITY PREDICTION USING PRE-COMPUTED SMOOTHED N-GRAMS ===")
    
    # Load pre-computed smoothed probabilities
    print("Loading pre-computed smoothed probabilities...")
    try:
        smoothed_data = load_smoothed_results()
        print("✓ Successfully loaded smoothed probabilities from", SMOOTHED_RESULTS_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {SMOOTHED_RESULTS_FILE}")
        print("Please ensure the smoothed results file exists in the correct location.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {SMOOTHED_RESULTS_FILE}")
        return
    
    # Load sentences from news articles
    print(f"\nLoading sentences from {NEWS_ARTICLE}...")
    sentences = get_sentences_from_articles(NEWS_ARTICLE, 1000)
    print(f"Loaded {len(sentences)} sentences")

    # Display information about available smoothing methods
    available_methods = list(smoothed_data["1_gram"]["smoothed_probabilities"].keys())
    print(f"Available smoothing methods: {available_methods}")
    
    # Get vocabulary information from smoothed data
    vocab_info = {}
    for n in [1, 2, 3, 4]:
        ngram_key = f"{n}_gram"
        if ngram_key in smoothed_data:
            vocab_info[ngram_key] = {
                'vocab_size': smoothed_data[ngram_key]["vocab_size"],
                'total_unique_ngrams': smoothed_data[ngram_key]["total_unique_ngrams"]
            }
            print(f"  {n}-gram: vocab_size={vocab_info[ngram_key]['vocab_size']}, unique_ngrams={vocab_info[ngram_key]['total_unique_ngrams']}")

    # For each sentence, compute probability using pre-computed smoothed probabilities
    print(f"\nComputing probabilities for {len(sentences)} sentences using pre-computed smoothed data...")
    results = []
    
    # Process sentences in batches for progress reporting
    batch_size = 100
    for i, sent in enumerate(sentences):
        if i % batch_size == 0:
            print(f"  Processed {i}/{len(sentences)} sentences...")
            
        tokens = sent.split()  # Simple tokenization
        sent_result = {'sentence': sent, 'token_count': len(tokens)}
        
        # Calculate probabilities for each n-gram order and smoothing method
        for n in [1, 2, 3, 4]:
            ngram_key = f"{n}_gram"
            if ngram_key not in smoothed_data:
                continue
                
            # Use available smoothing methods from the data
            for method in available_methods:
                try:
                    prob = calculate_sentence_probability_from_smoothed(
                        tokens, n, method, smoothed_data
                    )
                    sent_result[f'{n}gram_{method}'] = prob
                except Exception as e:
                    print(f"Warning: Error calculating {n}gram_{method} for sentence {i}: {e}")
                    sent_result[f'{n}gram_{method}'] = 0.0
            
        results.append(sent_result)

    print(f"✓ Processed all {len(sentences)} sentences")

    # Save results with enhanced information
    output_file = 'lab4/sentence_probabilities_from_smoothed.json'
    enhanced_results = {
        'metadata': {
            'total_sentences': len(sentences),
            'vocab_info': vocab_info,
            'smoothing_methods': available_methods,
            'ngram_orders': [1, 2, 3, 4],
            'source_file': SMOOTHED_RESULTS_FILE,
            'news_articles_file': NEWS_ARTICLE
        },
        'sentence_probabilities': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    
    # Display sample results
    print("\n📊 SAMPLE RESULTS:")
    print("-" * 80)
    sample_sentences = results[:3]  # Show first 3 sentences
    
    for i, sent_data in enumerate(sample_sentences, 1):
        print(f"\nSample {i}: '{sent_data['sentence'][:50]}{'...' if len(sent_data['sentence']) > 50 else ''}'")
        print(f"Tokens: {sent_data['token_count']}")
        print("Probabilities (1-gram):")
        if 'add_one' in available_methods:
            print(f"  Add-One: {sent_data.get('1gram_add_one', 0):.2e}")
        if 'token_type' in available_methods:
            print(f"  Token-Type: {sent_data.get('1gram_token_type', 0):.2e}")
        # Show first available method if add_one not available
        if available_methods and 'add_one' not in available_methods:
            method = available_methods[0]
            print(f"  {method}: {sent_data.get(f'1gram_{method}', 0):.2e}")
    
    print(f"\n📁 Complete results with metadata saved to: {output_file}")
    print("🔍 This data contains sentence probabilities computed from pre-computed smoothed models!")
    
    # Display summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print("-" * 40)
    if results:
        # Find sentences with highest and lowest probabilities for 1-gram add-one
        if available_methods and f'1gram_{available_methods[0]}' in results[0]:
            method_key = f'1gram_{available_methods[0]}'
            valid_results = [r for r in results if r.get(method_key, 0) > 0]
            if valid_results:
                highest_prob = max(valid_results, key=lambda x: x[method_key])
                lowest_prob = min(valid_results, key=lambda x: x[method_key])
                
                print(f"Highest probability sentence ({available_methods[0]}):")
                print(f"  '{highest_prob['sentence'][:60]}{'...' if len(highest_prob['sentence']) > 60 else ''}'")
                print(f"  Probability: {highest_prob[method_key]:.2e}")
                
                print(f"\nLowest probability sentence ({available_methods[0]}):")
                print(f"  '{lowest_prob['sentence'][:60]}{'...' if len(lowest_prob['sentence']) > 60 else ''}'")
                print(f"  Probability: {lowest_prob[method_key]:.2e}")

if __name__ == "__main__":
    main()