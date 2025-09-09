import os
import random

import json
from LanguageModels import build_ngram_model, get_vocab, get_ngram_vocab, get_sentence_ngrams
from smoothing import add_one_smoothing, add_k_smoothing, token_type_smoothing, sentence_probability_smoothed

# Path to a folder containing news articles in your language (as plain text files)
NEWS_ARTICLE = "Lab4/news_articles.txt"

def get_sentences_from_articles(file_path, num_sentences=1000):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        file_sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences.extend(file_sentences)
    random.shuffle(sentences)
    return sentences[:num_sentences]

def main():
    # Load sentences from news articles
    sentences = get_sentences_from_articles(NEWS_ARTICLE, 1000)

    # Load tokenized data from Lab1 (Assignment-1)
    with open('Lab1/tokenized_hindi_tokens.json', 'r', encoding='utf-8') as f:
        hindi_tokens_nested = json.load(f)
    # Flatten the list of lists of lists
    hindi_tokens = [token for document in hindi_tokens_nested for sentence in document for token in sentence]

    # Build n-gram models
    models = {}
    for n in [1, 2, 3, 4]:
        ngram_counts, _ = build_ngram_model(hindi_tokens, n)
        vocab = get_ngram_vocab(hindi_tokens, n)
        models[n] = {
            'counts': ngram_counts,
            'vocab_size': len(vocab)
        }

    # For each sentence, compute probability using each smoothing technique and n-gram model
    results = []
    for sent in sentences:
        tokens = sent.split()  # Simple tokenization
        sent_result = {'sentence': sent}
        for n in [1, 2, 3, 4]:
            model = models[n]
            # Add-One Smoothing
            prob_add_one = sentence_probability_smoothed(tokens, model['counts'], model['vocab_size'], add_one_smoothing, n)
            # Add-K Smoothing (k=0.5)
            prob_add_k = sentence_probability_smoothed(tokens, model['counts'], model['vocab_size'], add_k_smoothing, n, k=0.5)
            # Token Type Smoothing
            prob_type = sentence_probability_smoothed(tokens, model['counts'], model['vocab_size'], token_type_smoothing, n)
            sent_result[f'{n}gram_add_one'] = prob_add_one
            sent_result[f'{n}gram_add_k'] = prob_add_k
            sent_result[f'{n}gram_type'] = prob_type
        results.append(sent_result)

    output_file = 'Lab4/sentence_probabilities.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()