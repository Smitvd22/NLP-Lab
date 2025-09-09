"""
Simplified Interpolated Kneser-Ney implementation for n-gram orders up to 4.

Behavior:
- Loads TSV n-gram counts from Lab4
- Computes continuation counts and interpolation probabilities
- Exposes get_probability(ngram)
- Supports training and saving model

This implementation uses absolute discounting with an estimated discount per order.
"""
from collections import defaultdict, Counter
from pathlib import Path
import json
import math
import os
from typing import Tuple


class KneserNeyModel:
    def __init__(self, max_n: int = 4, discounts: dict = None):
        self.max_n = max_n
        self.ngram_counts = {n: defaultdict(int) for n in range(1, max_n + 1)}
        self.cont_counts = {n: defaultdict(int) for n in range(1, max_n + 1)}
        self.discounts = discounts or {1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75}
        self.vocab = set()
        self.total_unigrams = 0
        self.probs = {n: {} for n in range(1, max_n + 1)}

    def load_ngram_data(self, language: str = 'hindi'):
        base_path = Path(__file__).parent.parent / 'Lab4'
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
                        self.total_unigrams += count

        filenames = {2: f'{language}_bigrams.tsv', 3: f'{language}_trigrams.tsv', 4: f'{language}_quadragrams.tsv'}
        for n, filename in filenames.items():
            path = base_path / filename
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            key = parts[0]
                            try:
                                count = int(parts[1])
                            except ValueError:
                                continue
                            tokens = tuple(key.split()) if ' ' in key else tuple(key.split('|||'))
                            if len(tokens) != n:
                                tokens = tuple(key.split('|||'))
                            if len(tokens) != n:
                                continue
                            self.ngram_counts[n][tokens] = count
                            for t in tokens:
                                self.vocab.add(t)

    def compute_continuation_counts(self):
        # For each word (unigram) continuation count is number of unique contexts it appears in
        # For higher orders, cont_counts[n][ngram] = number of unique histories that see that continuation
        # We'll compute needed statistics for Kneser-Ney
        # unigrams continuation: number of unique bigram contexts that end with word
        forward_contexts = defaultdict(set)
        for bigram in self.ngram_counts[2]:
            w = bigram[1]
            forward_contexts[w].add(bigram[0])
        for w, s in forward_contexts.items():
            self.cont_counts[1][(w,)] = len(s)

        # for n > 1, continuation count of an n-gram is number of unique histories it appears after
        for n in range(2, self.max_n + 1):
            hist_sets = defaultdict(set)
            for ngram in self.ngram_counts[n]:
                continuation = ngram[-1]
                history = ngram[:-1]
                hist_sets[history].add(continuation)
            # we store continuation sizes per history for interpolation
            for history, s in hist_sets.items():
                self.cont_counts[n][history] = len(s)

    def estimate_discounts(self):
        # Simple heuristic: keep defaults or use Good-Turing-ish if needed
        for n in range(1, self.max_n + 1):
            counts = list(self.ngram_counts[n].values())
            if not counts:
                continue
            freq = Counter(counts)
            n1 = freq.get(1, 0)
            n2 = freq.get(2, 0)
            if n1 > 0 and n2 > 0:
                d = n1 / (n1 + 2 * n2)
                self.discounts[n] = max(0.5, min(0.95, d))

    def compute_probabilities(self):
        # Interpolated Kneser-Ney (simplified)
        self.compute_continuation_counts()
        self.estimate_discounts()

        # Unigram probabilities are continuation-based
        total_cont = sum(self.cont_counts[1].values()) if self.cont_counts[1] else max(1, sum(self.ngram_counts[1].values()))
        for unigram in self.cont_counts[1]:
            self.probs[1][unigram] = self.cont_counts[1][unigram] / total_cont

        # For higher orders compute discounted counts and interpolation
        for n in range(2, self.max_n + 1):
            D = self.discounts.get(n, 0.75)
            # Denominators: counts of histories
            history_counts = defaultdict(int)
            for ngram, c in self.ngram_counts[n].items():
                history = ngram[:-1]
                history_counts[history] += c

            for ngram, c in self.ngram_counts[n].items():
                history = ngram[:-1]
                cont = ngram[-1]
                discounted = max(c - D, 0.0)
                if history_counts[history] > 0:
                    p_cont = discounted / history_counts[history]
                else:
                    p_cont = 0.0
                self.probs[n][ngram] = p_cont

        # Note: for unseen continuation probabilities we will backoff to lower order during get_probability

    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        n = len(ngram)
        if n > self.max_n:
            ngram = ngram[-self.max_n:]
            n = len(ngram)

        # exact observed ngram
        if ngram in self.probs.get(n, {}):
            return self.probs[n][ngram]

        if n == 1:
            # unseen unigram -> small prob
            return 1.0 / max(1, len(self.vocab) * 1000)

        # Backoff: combine discounted higher-order mass with lower-order probability
        history = ngram[:-1]
        D = self.discounts.get(n, 0.75)
        # compute alpha(history) = (D * N1+(history, .)) / C(history)
        unique_cont = sum(1 for g in self.ngram_counts[n] if g[:-1] == history)
        C_hist = sum(c for g, c in self.ngram_counts[n].items() if g[:-1] == history)
        if C_hist > 0:
            alpha = (D * unique_cont) / C_hist
        else:
            alpha = 1.0

        lower_prob = self.get_probability(ngram[1:])
        return alpha * lower_prob

    def train(self, language: str = 'hindi'):
        self.load_ngram_data(language)
        self.compute_probabilities()

    def save_model(self, filepath: str):
        model = {
            'ngram_counts': {str(n): {"|||".join(k): v for k, v in self.ngram_counts[n].items()} for n in self.ngram_counts},
            'discounts': self.discounts,
            'vocab': list(self.vocab),
            'max_n': self.max_n
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    def load_model(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            model = json.load(f)
        self.max_n = model.get('max_n', self.max_n)
        self.discounts = model.get('discounts', self.discounts)
        self.vocab = set(model.get('vocab', []))
        self.ngram_counts = {int(n): defaultdict(int, {tuple(k.split('|||')): v for k, v in d.items()}) for n, d in model['ngram_counts'].items()}


def main(languages=None, num_sentences: int = 100):
    """Train Kneser-Ney models for requested languages and generate sentences.
    """
    languages = languages or ['hindi', 'marathi']
    try:
        from Lab6.sentence_generation import generate_sentences_for_model
    except Exception:
        from sentence_generation import generate_sentences_for_model

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"Training Kneser-Ney model (max_n=4) for language: {lang}")
        model = KneserNeyModel(max_n=4)
        model.train(lang)
        model_path = results_dir / f'kneser_ney_{lang}_model.json'
        model.save_model(str(model_path))
        print(f"Saved Kneser-Ney model to {model_path}")
        print(f"Generating {num_sentences} sentences (greedy + beam) for {lang}")
        generate_sentences_for_model(model, 'Kneser-Ney', lang, num_sentences=num_sentences)


if __name__ == '__main__':
    main()