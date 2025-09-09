"""
Katz Backoff implementation for quadrigram models (orders 1..4).

This is a practical, self-contained implementation that:
- Loads n-gram TSV files produced in Lab4: <language>_unigrams.tsv, _bigrams.tsv, _trigrams.tsv, _quadragrams.tsv
- Computes Good-Turing discounts for low counts (simple implementation)
- Produces backoff probabilities and exposes get_probability(ngram)
- Supports saving/loading model JSON

Notes:
- This implementation focuses on correctness and reasonable defaults rather than extreme optimization.
"""
from collections import defaultdict, Counter
from pathlib import Path
import json
import math
import os
from typing import Tuple


class KatzBackoffModel:
    def __init__(self, max_n: int = 4, discount_threshold: int = 5):
        self.max_n = max_n
        self.discount_threshold = discount_threshold
        self.ngram_counts = {n: defaultdict(int) for n in range(1, max_n + 1)}
        self.ngram_probs = {n: {} for n in range(1, max_n + 1)}
        self.backoff_weights = {n: {} for n in range(1, max_n)}
        self.vocab = set()
        self.total_tokens = 0

    def _parse_tuple_key(self, key: str) -> Tuple[str, ...]:
        return tuple(key.split('|||')) if key else tuple()

    def load_ngram_data(self, language: str = 'hindi'):
        base_path = Path(__file__).parent.parent / 'Lab4'

        # Unigrams
        unigrams_file = base_path / f'{language}_unigrams.tsv'
        if unigrams_file.exists():
            with open(unigrams_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        try:
                            count = int(parts[1])
                        except ValueError:
                            continue
                        self.ngram_counts[1][(token,)] = count
                        self.vocab.add(token)
                        self.total_tokens += count

        # higher-order n-grams
        filenames = {2: f'{language}_bigrams.tsv', 3: f'{language}_trigrams.tsv', 4: f'{language}_quadragrams.tsv'}
        for n, filename in filenames.items():
            path = base_path / filename
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            ngram = tuple(parts[0].split()) if ' ' in parts[0] else tuple(parts[0].split('|||'))
                            try:
                                count = int(parts[1])
                            except ValueError:
                                continue
                            # normalize to tuple of tokens
                            if len(ngram) != n:
                                # try separating by '|||'
                                ngram = tuple(parts[0].split('|||'))
                            if len(ngram) != n:
                                continue
                            self.ngram_counts[n][ngram] = count
                            for t in ngram:
                                self.vocab.add(t)

    def good_turing_discount(self, counts_dict):
        freq_of_freq = Counter(counts_dict.values())
        discounted = {}
        for ngram, c in counts_dict.items():
            if c <= 0:
                discounted_count = 0.0
            elif c <= self.discount_threshold and freq_of_freq.get(c, 0) > 0:
                nc = freq_of_freq.get(c, 0)
                nc1 = freq_of_freq.get(c + 1, 0)
                if nc1 > 0:
                    discounted_count = (c + 1) * (nc1 / nc)
                else:
                    discounted_count = max(c - 0.5, 0.0)
            else:
                discounted_count = float(c)
            discounted[ngram] = max(discounted_count, 0.0)
        return discounted

    def compute_backoff_weights(self):
        # For each context, backoff weight alpha(context) = 1 - sum(P_disc(w|context)) over observed continuations
        for n in range(2, self.max_n + 1):
            # group by context
            context_groups = defaultdict(list)
            for ngram, prob in self.ngram_probs[n].items():
                context = ngram[:-1]
                context_groups[context].append((ngram[-1], prob))

            for context, cont_list in context_groups.items():
                seen_prob = sum(p for _, p in cont_list)
                self.backoff_weights[n - 1][context] = max(1.0 - seen_prob, 0.0)

    def compute_probabilities(self):
        # Unigram probabilities (MLE)
        if self.total_tokens == 0 and self.ngram_counts[1]:
            self.total_tokens = sum(self.ngram_counts[1].values())
        for unigram, count in self.ngram_counts[1].items():
            self.ngram_probs[1][unigram] = count / max(1, self.total_tokens)

        # Higher order
        for n in range(2, self.max_n + 1):
            discounted = self.good_turing_discount(self.ngram_counts[n])
            # For each ngram, compute discounted count / context_count
            context_counts = defaultdict(float)
            for ngram, dc in discounted.items():
                context = ngram[:-1]
                context_counts[context] += dc

            for ngram, dc in discounted.items():
                context = ngram[:-1]
                denom = context_counts.get(context, 0.0)
                if denom > 0:
                    self.ngram_probs[n][ngram] = dc / denom
                else:
                    self.ngram_probs[n][ngram] = 0.0

        # compute backoff weights
        self.compute_backoff_weights()

    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        n = len(ngram)
        if n > self.max_n:
            # use suffix of size max_n
            ngram = ngram[-self.max_n:]
            n = len(ngram)

        if ngram in self.ngram_probs.get(n, {}):
            return self.ngram_probs[n][ngram]

        if n == 1:
            # unseen unigram -> small smoothing
            return 1.0 / max(1, len(self.vocab) * 1000)

        context = ngram[:-1]
        backoff_weight = self.backoff_weights[n - 1].get(context, 1.0)
        lower_prob = self.get_probability(ngram[1:])
        return backoff_weight * lower_prob

    def train(self, language: str = 'hindi'):
        self.load_ngram_data(language)
        # ensure total_tokens
        if self.total_tokens == 0 and self.ngram_counts[1]:
            self.total_tokens = sum(self.ngram_counts[1].values())
        self.compute_probabilities()

    def save_model(self, filepath: str):
        model = {
            'ngram_counts': {str(n): {"|||".join(k): v for k, v in self.ngram_counts[n].items()} for n in self.ngram_counts},
            'ngram_probs': {str(n): {"|||".join(k): v for k, v in self.ngram_probs[n].items()} for n in self.ngram_probs},
            'backoff_weights': {str(n): {"|||".join(k): v for k, v in self.backoff_weights[n].items()} for n in self.backoff_weights},
            'vocab': list(self.vocab),
            'total_tokens': self.total_tokens,
            'max_n': self.max_n
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    def load_model(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            model = json.load(f)
        self.max_n = model.get('max_n', self.max_n)
        self.total_tokens = model.get('total_tokens', self.total_tokens)
        self.vocab = set(model.get('vocab', []))
        self.ngram_counts = {int(n): defaultdict(int, {tuple(k.split('|||')): v for k, v in d.items()}) for n, d in model['ngram_counts'].items()}
        self.ngram_probs = {int(n): {tuple(k.split('|||')): v for k, v in d.items()} for n, d in model['ngram_probs'].items()}
        self.backoff_weights = {int(n): {tuple(k.split('|||')): v for k, v in d.items()} for n, d in model['backoff_weights'].items()}


def main(languages=None, num_sentences: int = 100):
    """Train Katz Backoff models for requested languages and generate sentences.

    Saves model JSON files under Lab6/results/ and generates sentences using
    the utilities in sentence_generation.generate_sentences_for_model.
    """
    languages = languages or ['hindi', 'marathi']
    # import here to avoid circular imports when module is imported elsewhere
    try:
        from Lab6.sentence_generation import generate_sentences_for_model
    except Exception:
        # fallback local import when running from Lab6 directory
        from sentence_generation import generate_sentences_for_model

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"Training Katz Backoff model (max_n=4) for language: {lang}")
        model = KatzBackoffModel(max_n=4)
        model.train(lang)
        model_path = results_dir / f'katz_backoff_{lang}_model.json'
        model.save_model(str(model_path))
        print(f"Saved Katz model to {model_path}")
        print(f"Generating {num_sentences} sentences (greedy + beam) for {lang}")
        generate_sentences_for_model(model, 'Katz Backoff', lang, num_sentences=num_sentences)


if __name__ == '__main__':
    # default behavior when executed directly: train both languages and generate 100 sentences
    main()