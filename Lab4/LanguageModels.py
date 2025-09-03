"""
Lab 4 - Simple N‑gram Models (Unigram, Bigram, Trigram, Quadragram)

Requirement adjustments:
 - Memory-friendly: stream tokens instead of loading whole list; maintain rolling window.
 - Outputs: unigrams.tsv, bigrams.tsv, trigrams.tsv, quadragrams.tsv
   * Unigrams: token \t count \t p(token)
   * Higher n: w1..wn \t count \t p(last|history)
 - Prints top 10 most frequent n‑grams for each order.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque

MAX_N = 4
TOP_PRINT = 10
MAX_UNIQUE_PER_ORDER = None  # e.g., 500000 to cap memory

def load_tokens(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		return json.load(f)

def stream_tokens_from_json(tokens_list):
	"""Stream tokens from a flattened list of tokens"""
	for tok in tokens_list:
		if tok.strip():
			yield tok.strip()

def unigram_prob(count: int, total: int) -> float:
	return count / total if total else 0.0

def conditional_prob(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int]) -> float:
	history = ngram[:-1]
	denom = counts_prev.get(history, 0) # frequency of history
	return counts_n.get(ngram, 0) / denom if denom else 0.0 # P(w_n | history)

def write_unigrams(counts: Dict[Tuple[str, ...], int], total: int, out_path: Path):
	rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
	with out_path.open("w", encoding="utf-8") as f:
		f.write("token\tcount\tp\n")
		for (tok,), c in rows:
			f.write(f"{tok}\t{c}\t{unigram_prob(c, total):.8f}\n")

def write_higher(n: int, counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int], out_path: Path):
	rows = sorted(counts_n.items(), key=lambda kv: (-kv[1], kv[0]))
	header = [f"w{i+1}" for i in range(n)] + ["count", "p_cond"]
	with out_path.open("w", encoding="utf-8") as f:
		f.write("\t".join(header) + "\n")
		for gram, c in rows:
			p = conditional_prob(gram, counts_n, counts_prev)
			f.write("\t".join(list(gram) + [str(c), f"{p:.8f}"]) + "\n")

def print_top(counts: Dict[Tuple[str, ...], int], n: int):
	rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_PRINT]
	print(f"Top {len(rows)} {n}-grams:")
	for gram, c in rows:
		print(f"  {' '.join(gram):<60} {c}")
	print()

def build_ngram_model(tokens_list, n):
	"""Build n-gram model from a list of tokens"""
	counts = defaultdict(int)
	vocab = set()
	
	# Convert tokens to list if it's a generator
	if hasattr(tokens_list, '__iter__') and not isinstance(tokens_list, (list, tuple)):
		tokens_list = list(tokens_list)
	
	for token in tokens_list:
		vocab.add(token)
	
	# Generate n-grams
	for i in range(len(tokens_list) - n + 1):
		ngram = tuple(tokens_list[i:i + n])
		counts[ngram] += 1
	
	return counts, vocab

def get_vocab(tokens_list):
	"""Get vocabulary from tokens"""
	return set(tokens_list)

def get_ngram_vocab(tokens_list, n):
	"""Get n-gram vocabulary from tokens"""
	vocab = set()
	for i in range(len(tokens_list) - n + 1):
		ngram = tuple(tokens_list[i:i + n])
		vocab.add(ngram)
	return vocab

def get_sentence_ngrams(sentence_tokens, n):
	"""Get n-grams from a sentence"""
	ngrams = []
	for i in range(len(sentence_tokens) - n + 1):
		ngram = tuple(sentence_tokens[i:i + n])
		ngrams.append(ngram)
	return ngrams

def process_language(language_name, tokens_list):
	"""Process tokens for a specific language using streaming approach"""
	print(f"\n=== {language_name} Language Models ===")
	print(f"Streaming tokens from: {language_name} data")

	counts: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, MAX_N + 1)}
	total_tokens = 0
	vocab = set()
	window: Deque[str] = deque(maxlen=MAX_N - 1)

	for tok in stream_tokens_from_json(tokens_list):
		total_tokens += 1
		vocab.add(tok)
		counts[1][(tok,)] += 1
		if MAX_N > 1:
			hist = list(window)
			hl = len(hist) 
			for n in range(2, MAX_N + 1):
				need = n - 1 
				if hl >= need:
					gram = tuple(hist[-need:] + [tok])
					counts[n][gram] += 1
					
		window.append(tok)

	print(f"Total tokens: {total_tokens}; Vocabulary size: {len(vocab)}")
	for n in range(1, MAX_N + 1):
		print(f"Unique {n}-grams: {len(counts[n])}")

	out_dir = Path(__file__).parent
	language_prefix = language_name.lower()
	
	# Write outputs
	write_unigrams(counts[1], total_tokens, out_dir / f"{language_prefix}_unigrams.tsv")
	print_top(counts[1], 1)

	file_names = {2: "bigrams.tsv", 3: "trigrams.tsv", 4: "quadragrams.tsv"}
	for n in range(2, MAX_N + 1):
		write_higher(n, counts[n], counts[n - 1], out_dir / f"{language_prefix}_{file_names[n]}")
		print_top(counts[n], n)

if __name__ == "__main__":
	# Load Hindi and Marathi tokens
	hindi_tokens_nested = load_tokens('Lab1/tokenized_hindi_tokens.json')
	marathi_tokens_nested = load_tokens('Lab1/tokenized_marathi_tokens.json')
	# Flatten the list of lists of lists
	hindi_tokens = [token for document in hindi_tokens_nested for sentence in document for token in sentence]
	marathi_tokens = [token for document in marathi_tokens_nested for sentence in document for token in sentence]

	# Process Hindi
	process_language("Hindi", hindi_tokens)
	
	# Process Marathi
	process_language("Marathi", marathi_tokens)

