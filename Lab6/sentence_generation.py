"""
Sentence generation utilities for n-gram models.

Provides:
- SentenceGenerator: greedy and beam search generation using a supplied model that implements get_probability(ngram)
- generate_sentences_for_model: convenience function to generate, evaluate and save results in Lab6/results/
"""
from typing import List, Tuple, Dict
from collections import defaultdict
import math
import random
import json
import os
from pathlib import Path


class SentenceGenerator:
    def __init__(self, model, max_length: int = 20, min_length: int = 3):
        self.model = model
        self.max_length = max_length
        self.min_length = min_length
        self.vocab = sorted(list(getattr(model, 'vocab', [])))
        # ensure tokens
        self.start_token = '<s>'
        self.end_token = '</s>'
        if self.start_token not in self.vocab:
            self.vocab.append(self.start_token)
        if self.end_token not in self.vocab:
            self.vocab.append(self.end_token)

    def _next_word_probs(self, context: Tuple[str, ...]) -> Dict[str, float]:
        probs = {}
        for w in self.vocab:
            if w == self.start_token:
                continue
            ngram = tuple(list(context) + [w])
            p = self.model.get_probability(ngram)
            probs[w] = p
        # normalize
        s = sum(probs.values())
        if s <= 0:
            uniform = 1.0 / max(1, len(probs))
            return {w: uniform for w in probs}
        return {w: p / s for w, p in probs.items()}

    def greedy_generation(self, num_sentences: int = 100, seed: int = None) -> List[str]:
        if seed is not None:
            random.seed(seed)
        sentences = []
        for _ in range(num_sentences):
            sent = []
            context = tuple()
            for _ in range(self.max_length):
                probs = self._next_word_probs(context)
                # choose argmax
                next_word = max(probs.items(), key=lambda x: x[1])[0]
                # stop if end token chosen and we've reached minimum length
                if next_word == self.end_token:
                    if len(sent) >= self.min_length:
                        break
                    # otherwise pick next best (avoid immediate end)
                    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    # find first non-end token
                    next_word = None
                    for w, _ in sorted_items:
                        if w != self.end_token:
                            next_word = w
                            break
                    if next_word is None:
                        break
                sent.append(next_word)
                # update context keep last (model.max_n-1) tokens if available
                if hasattr(self.model, 'max_n'):
                    k = max(1, self.model.max_n - 1)
                    context = tuple((list(context) + [next_word])[-k:])
                else:
                    context = tuple((list(context) + [next_word])[-3:])
            sentences.append(' '.join(sent))
        return sentences

    def beam_search_generation(self, num_sentences: int = 100, beam_size: int = 20, seed: int = None) -> List[str]:
        if seed is not None:
            random.seed(seed)
        results = []
        for _ in range(num_sentences):
            # beam contains tuples (logprob, tokens, context)
            beam = [(0.0, [], tuple())]
            completed = []
            for _ in range(self.max_length):
                new_beam = []
                for logp, tokens, context in beam:
                    probs = self._next_word_probs(context)
                    # take top K expansions
                    top_k = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:beam_size]
                    for w, p in top_k:
                        if p <= 0:
                            continue
                        new_tokens = tokens + [w]
                        new_logp = logp + math.log(p)
                        if w == self.end_token:
                            completed.append((new_logp, new_tokens))
                        else:
                            if hasattr(self.model, 'max_n'):
                                k = max(1, self.model.max_n - 1)
                                new_context = tuple((list(context) + [w])[-k:])
                            else:
                                new_context = tuple((list(context) + [w])[-3:])
                            new_beam.append((new_logp, new_tokens, new_context))
                # prune
                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_size]
                if not beam:
                    break
            if completed:
                best = max(completed, key=lambda x: x[0])[1]
                # remove end token if present
                if best and best[-1] == self.end_token:
                    best = best[:-1]
                results.append(' '.join([t for t in best if t not in (self.start_token, self.end_token)]))
            else:
                # fall back to best partial
                if beam:
                    best_partial = max(beam, key=lambda x: x[0])[1]
                    results.append(' '.join([t for t in best_partial if t not in (self.start_token, self.end_token)]))
                else:
                    results.append('')
        return results


def generate_sentences_for_model(model, model_name: str, language: str, num_sentences: int = 100):
    gen = SentenceGenerator(model, max_length=15, min_length=3)
    greedy = gen.greedy_generation(num_sentences=num_sentences, seed=42)
    beam = gen.beam_search_generation(num_sentences=num_sentences, beam_size=20, seed=42)

    out = {
        'model_name': model_name,
        'language': language,
        'num_sentences': num_sentences,
        'greedy': greedy,
        'beam_search': beam
    }
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{model_name.lower().replace(' ', '_')}_{language}_generated_sentences.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # also save readable text files
    base = results_dir / f"{model_name.lower().replace(' ', '_')}_{language}"
    with open(base.with_name(base.name + '_greedy_sentences.txt'), 'w', encoding='utf-8') as f:
        for s in greedy:
            f.write(s + '\n')
    with open(base.with_name(base.name + '_beam_search_sentences.txt'), 'w', encoding='utf-8') as f:
        for s in beam:
            f.write(s + '\n')
    return out