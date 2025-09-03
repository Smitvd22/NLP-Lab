import os
from collections import defaultdict

BRANCH_THRESHOLD = 15  # Hyperparameter, can be tuned

class TrieNode:
	def __init__(self):
		self.children = {}
		self.cnt = 0

def insert_word(root, word):
	node = root
	node.cnt += 1
	for c in word:
		if c not in node.children:
			node.children[c] = TrieNode()
		node = node.children[c]
		node.cnt += 1

def best_split(root, word):
	node = root
	best_i = -1
	best_score = 0.0
	best_support = 0
	for i, c in enumerate(word):
		if c not in node.children:
			break
		node = node.children[c]
		branching = len(node.children)
		if branching < BRANCH_THRESHOLD:
			continue
		max_child = max((child.cnt for child in node.children.values()), default=0)
		if node.cnt <= 0:
			continue
		frac = 1.0 - float(max_child) / float(node.cnt)
		score = frac * branching
		if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and i > best_i):
			best_i = i
			best_score = score
			best_support = node.cnt
	return best_i, best_score, best_support

def best_split_suffix(root, word):
	node = root
	best_i = -1
	best_score = 0.0
	best_support = 0
	for i, c in enumerate(reversed(word)):
		if c not in node.children:
			break
		node = node.children[c]
		branching = len(node.children)
		if branching < BRANCH_THRESHOLD:
			continue
		max_child = max((child.cnt for child in node.children.values()), default=0)
		if node.cnt <= 0:
			continue
		frac = 1.0 - float(max_child) / float(node.cnt)
		score = frac * branching
		if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and i > best_i):
			best_i = i
			best_score = score
			best_support = node.cnt
	return best_i, best_score, best_support

def load_words(path):
	with open(path, 'r', encoding='utf-8') as f:
		return [line.strip().lower() for line in f if line.strip()]



# Returns (stem, prefix) if a common prefix is found, else None
def looks_common_prefix(s):
	prefixes = [
		("un", lambda s: s[2:]),
		("re", lambda s: s[2:]),
		("in", lambda s: s[2:]),
		("im", lambda s: s[2:]),
		("dis", lambda s: s[3:]),
		("en", lambda s: s[2:]),
		("non", lambda s: s[3:]),
		("over", lambda s: s[4:]),
		("mis", lambda s: s[3:]),
		("sub", lambda s: s[3:]),
		("pre", lambda s: s[3:]),
		("inter", lambda s: s[5:]),
		("fore", lambda s: s[4:]),
		("de", lambda s: s[2:]),
		("trans", lambda s: s[5:]),
		("super", lambda s: s[5:]),
		("semi", lambda s: s[4:]),
		("anti", lambda s: s[4:]),
		("mid", lambda s: s[3:]),
		("under", lambda s: s[5:]),
	]
	for pre, stem_func in prefixes:
		if len(s) > len(pre)+1 and s.startswith(pre):
			return pre, stem_func(s)
	return None

def looks_common_suffix(s):
	# Returns (stem, suffix) if a common suffix is found, else None
	suffixes = [
		("ies", lambda s: s[:-3] + "y"),
		("es", lambda s: s[:-2]),
		("s", lambda s: s[:-1]),
		("ing", lambda s: s[:-3]),
		("ed", lambda s: s[:-2]),
		("er", lambda s: s[:-2]),
		("est", lambda s: s[:-3]),
		("ment", lambda s: s[:-4]),
		("ness", lambda s: s[:-4]),
		("tion", lambda s: s[:-4]),
		("able", lambda s: s[:-4]),
		("ful", lambda s: s[:-3]),
		("al", lambda s: s[:-2]),
		("ous", lambda s: s[:-3]),
		("ly", lambda s: s[:-2]),
	]
	for suf, stem_func in suffixes:
		if len(s) > len(suf)+1 and s.endswith(suf):
			return stem_func(s), suf
	return None

def main():
	base_path = r'c:\Users\acer\Desktop\U23AI118\SEM 5\NLP-Lab'
	nouns_path = os.path.join(base_path, 'Lab2', 'brown_nouns.txt')
	words = load_words(nouns_path)
	total_words = len(words)
	# Precompute word frequencies for fast lookup
	word_freq = defaultdict(int)
	for w in words:
		word_freq[w] += 1

	pref = TrieNode()
	suf = TrieNode()
	for w in words:
		insert_word(pref, w)
		insert_word(suf, w[::-1])

	prefix_out_path = os.path.join(base_path, 'Lab3', 'prefix_out.txt')
	suffix_out_path = os.path.join(base_path, 'Lab3', 'suffix_out.txt')
	pref_count = 0
	suf_count = 0
	pref_score_sum = 0.0
	suf_score_sum = 0.0

	with open(prefix_out_path, 'w', encoding='utf-8') as pref_ofs, open(suffix_out_path, 'w', encoding='utf-8') as suf_ofs:

		for w in words:
			# Prefix Trie Stemming
			common_p = looks_common_prefix(w)
			if common_p:
				pre, stem_p = common_p
				score_p = 0
				support_pref = word_freq.get(stem_p, 0)
				prob_p = support_pref / total_words if total_words else 0
				pref_ofs.write(f'{w}={pre}+{stem_p} [freq={support_pref}, prob={prob_p:.4f}, common_prefix]\n')
				pref_count += 1
			else:
				ip, sp, support_p = best_split(pref, w)
				stem_p = ""
				sfx_p = ""
				score_p = sp
				support_pref = support_p
				prob_p = support_pref / total_words if total_words else 0
				if ip != -1:
					stem_p = w[:ip+1]
					sfx_p = w[ip+1:]
					if len(stem_p) < 2 or not sfx_p:
						stem_p = w
						sfx_p = ""
						score_p = 0
						support_pref = 0
						prob_p = 0
				else:
					stem_p = w
					sfx_p = ""
					score_p = 0
					support_pref = 0
					prob_p = 0
				if sfx_p:
					pref_ofs.write(f'{w}={stem_p}+{sfx_p} [score={score_p:.4f}, freq={support_pref}, prob={prob_p:.4f}]\n')
					pref_count += 1
					pref_score_sum += score_p
				else:
					pref_ofs.write(f'{w}={w}+ [nosplit]\n')

			# Suffix Trie Stemming
			common_s = looks_common_suffix(w)
			if common_s:
				stem_s, sfx_s = common_s
				score_s = 0
				support_suf = word_freq.get(stem_s, 0)
				prob_s = support_suf / total_words if total_words else 0
				suf_ofs.write(f'{w}={stem_s}+{sfx_s} [freq={support_suf}, prob={prob_s:.4f}, common_suffix]\n')
				suf_count += 1
			else:
				rw = w[::-1]
				ir, sr, support_s = best_split_suffix(suf, rw)
				stem_s = ""
				sfx_s = ""
				score_s = sr
				support_suf = support_s
				prob_s = support_suf / total_words if total_words else 0
				if ir != -1:
					rev = rw[:ir+1]
					sfx_s = rev[::-1]
					stem_s = w[:len(w)-len(sfx_s)]
					if len(stem_s) < 2 or not sfx_s:
						stem_s = w
						sfx_s = ""
						score_s = 0
						support_suf = 0
						prob_s = 0
				else:
					stem_s = w
					sfx_s = ""
					score_s = 0
					support_suf = 0
					prob_s = 0
				if sfx_s:
					suf_ofs.write(f'{w}={stem_s}+{sfx_s} [score={score_s:.4f}, freq={support_suf}, prob={prob_s:.4f}]\n')
					suf_count += 1
					suf_score_sum += score_s
				else:
					suf_ofs.write(f'{w}={w}+ [nosplit]\n')

	# Summary and winner
	winner = "prefix"
	if suf_count > pref_count:
		winner = "suffix"
	elif suf_count == pref_count:
		if suf_score_sum > pref_score_sum:
			winner = "suffix"

	chosen_fname = prefix_out_path if winner == "prefix" else suffix_out_path
	final_out_path = os.path.join(base_path, 'Lab3', 'trie_q1_output.txt')
	with open(chosen_fname, 'r', encoding='utf-8') as chosen_in, open(final_out_path, 'w', encoding='utf-8') as final_ofs:
		for l in chosen_in:
			final_ofs.write(l)
		final_ofs.write(f'\n---\nSummary:\nPrefix splits: {pref_count}, Suffix splits: {suf_count}\nPrefix score sum: {pref_score_sum:.4f}, Suffix score sum: {suf_score_sum:.4f}\nWinner: {winner}\n')

	print(f'written prefix_out={pref_count} suffix_out={suf_count} winner={winner}')

if __name__ == '__main__':
	main()
