import os
import json
import matplotlib.pyplot as plt

def load_tokens(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	# Flatten nested lists
	tokens = []
	for item in data:
		if isinstance(item, list):
			for subitem in item:
				if isinstance(subitem, list):
					tokens.extend(subitem)
				else:
					tokens.append(subitem)
		else:
			tokens.append(item)
	return tokens

def frequency_distribution(tokens):
	freq = {}
	for token in tokens:
		freq[token] = freq.get(token, 0) + 1
	return freq

def plot_freq_dist(freq, title, top_n=100, save_path=None):
	sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
	words, counts = zip(*sorted_items)
	plt.figure(figsize=(16,6))
	plt.bar(words, counts)
	# Set Devanagari-compatible font for x-axis labels
	font_properties = {'fontname': 'Nirmala UI'}
	plt.xticks(rotation=90, fontname='Nirmala UI')
	plt.title(title, fontname='Nirmala UI')
	plt.xlabel('Words', fontname='Nirmala UI')
	plt.ylabel('Frequency', fontname='Nirmala UI')
	if save_path:
		plt.savefig(save_path)
	plt.show()

def find_stop_words(freq, threshold):
	return {word for word, count in freq.items() if count >= threshold}

def main():
	base_path = r'c:\Users\acer\Desktop\U23AI118\SEM 5\NLP-Lab'
	tokens_path = os.path.join(base_path, 'Lab1', 'tokenized_hindi_tokens.json')
	tokens = load_tokens(tokens_path)
	freq = frequency_distribution(tokens)

	# Plot top 100 words
	plot_freq_dist(freq, 'Top 100 Most Frequent Words', top_n=100)

	# Identify stop words by frequency threshold
	thresholds = [25, 50, 100]  # Example thresholds, adjust as needed
	for threshold in thresholds:
		stop_words = find_stop_words(freq, threshold)
		filtered_freq = {word: count for word, count in freq.items() if word not in stop_words}
		plot_freq_dist(filtered_freq, f'Frequency Distribution after Removing Stop Words (threshold={threshold})', top_n=100)

if __name__ == '__main__':
	main()
