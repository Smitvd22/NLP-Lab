import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.freq += 1
        node.is_end = True

    def find_stem_suffix(self, word):
        node = self.root
        stem = ''
        max_branch = 0
        branch_idx = 0
        for i, char in enumerate(word):
            node = node.children[char]
            if len(node.children) > max_branch:
                max_branch = len(node.children)
                branch_idx = i + 1
            stem += char
        stem = word[:branch_idx]
        suffix = word[branch_idx:]
        return stem, suffix, node.freq

class SuffixTrie(Trie):
    def insert(self, word):
        node = self.root
        for char in reversed(word):
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.freq += 1
        node.is_end = True

    def find_stem_suffix(self, word):
        node = self.root
        stem = ''
        max_branch = 0
        branch_idx = 0
        for i, char in enumerate(reversed(word)):
            node = node.children[char]
            if len(node.children) > max_branch:
                max_branch = len(node.children)
                branch_idx = i + 1
            stem += char
        stem = word[len(word)-branch_idx:]
        suffix = word[:len(word)-branch_idx]
        return stem, suffix, node.freq

base_path = r'c:\Users\acer\Desktop\U23AI118\SEM 5\NLP-Lab'
nouns_path = os.path.join(base_path, 'Lab2', 'brown_nouns.txt')
with open(nouns_path, 'r', encoding='utf-8') as f:
    nouns = [line.strip() for line in f if line.strip()]

# Prepare output file
output_path = os.path.join(base_path, 'Lab3', 'output')
out_lines = []

prefix_trie = Trie()
suffix_trie = SuffixTrie()
for word in nouns:
    prefix_trie.insert(word)
    suffix_trie.insert(word)

out_lines.append('Prefix Trie Stemming:')
prefix_results = []
for word in nouns:
    stem, suffix, freq = prefix_trie.find_stem_suffix(word)
    if suffix:
        out_lines.append(f'{word}={stem}+{suffix} (freq={freq})')
    else:
        out_lines.append(f'{word}={stem} (freq={freq})')
    prefix_results.append((word, stem, suffix, freq))

out_lines.append('\nSuffix Trie Stemming:')
suffix_results = []
for word in nouns:
    stem, suffix, freq = suffix_trie.find_stem_suffix(word)
    if suffix:
        out_lines.append(f'{word}={stem}+{suffix} (freq={freq})')
    else:
        out_lines.append(f'{word}={stem} (freq={freq})')
    suffix_results.append((word, stem, suffix, freq))

def analyze_trie(results):
    suffix_lengths = [len(suf) for _, _, suf, _ in results if suf]
    avg_suffix = sum(suffix_lengths) / len(suffix_lengths) if suffix_lengths else 0
    return avg_suffix

avg_prefix = analyze_trie(prefix_results)
avg_suffix = analyze_trie(suffix_results)
out_lines.append(f'\nAverage suffix length (Prefix Trie): {avg_prefix:.2f}')
out_lines.append(f'Average suffix length (Suffix Trie): {avg_suffix:.2f}')
if avg_prefix < avg_suffix:
    out_lines.append('Prefix Trie is better for stemming.')
else:
    out_lines.append('Suffix Trie is better for stemming.')

try:
    with open(os.path.join(base_path, 'Lab1', 'tokenized_hindi_tokens.json'), 'r', encoding='utf-8') as f:
        tokens = json.load(f)
except:
    with open(os.path.join(base_path, 'Lab1', 'tokenized_marathi_tokens.json'), 'r', encoding='utf-8') as f:
        tokens = json.load(f)

def flatten_tokens(toks):
    flat = []
    for t in toks:
        if isinstance(t, list):
            flat.extend(flatten_tokens(t))
        else:
            flat.append(t)
    return flat

tokens = flatten_tokens(tokens)

freq_dist = defaultdict(int)
for token in tokens:
    freq_dist[token] += 1

top_100 = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:100]
words, freqs = zip(*top_100)
plt.figure(figsize=(16,6))
plt.bar(words, freqs)
plt.xticks(rotation=90)
plt.title('Top 100 Most Frequent Words')
plt.tight_layout()
plt.show()

thresholds = [10, 20, 50]
stop_words = set([w for w, f in freq_dist.items() if f > thresholds[0]])

def plot_after_stopwords(thresh):
    filtered = {w: f for w, f in freq_dist.items() if f <= thresh}
    top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:100]
    if not top:
        out_lines.append(f'No words above threshold {thresh}')
        return
    words, freqs = zip(*top)
    plt.figure(figsize=(16,6))
    plt.bar(words, freqs)
    plt.xticks(rotation=90)
    plt.title(f'Top 100 Words After Removing Stop Words (Threshold={thresh})')
    plt.tight_layout()
    plt.show()
    out_lines.append(f'Top 100 words after removing stop words (threshold={thresh}):')
    out_lines.extend([f'{w}: {f}' for w, f in top])

for thresh in thresholds:
    plot_after_stopwords(thresh)

with open(output_path, 'w', encoding='utf-8') as fout:
    for line in out_lines:
        fout.write(line + '\n')