import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import os

class BPETokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = {}
        self.vocab = set()
        
    def pre_tokenize(self, text: str) -> List[str]:
        """Enhanced pre-tokenization for multilingual text including Hindi and Marathi"""
        if not text or not text.strip():
            return []
        
        # Enhanced patterns for Indian languages
        patterns = [
            r'[\u0900-\u097F]+',  # Devanagari (Hindi, Marathi)
            r'[a-zA-Z]+',         # English words
            r'\d+',               # Numbers
            r'[^\w\s]'            # Punctuation
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        matches = re.findall(combined_pattern, text)
        
        # Flatten and filter
        words = []
        for match in matches:
            for group in match:
                if group and group.strip():
                    words.append(group.lower())
        
        return words
    
    def get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent symbol pairs"""
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            if word in splits:
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_symbols(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair in all words"""
        new_splits = {}
        first, second = pair
        
        for word in splits:
            symbols = splits[word]
            new_symbols = []
            i = 0
            
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == first and 
                    symbols[i + 1] == second):
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            
            new_splits[word] = new_symbols
        
        return new_splits
    
    def train(self, corpus: List[str], num_merges: int = 32000):
        """Train BPE on the corpus"""
        print(f"Training BPE with {num_merges} merges...")
        
        # Pre-tokenize and count word frequencies
        all_words = []
        for text in corpus:
            words = self.pre_tokenize(text)
            all_words.extend(words)
        
        self.word_freqs = Counter(all_words)
        print(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize splits with character-level segmentation
        self.splits = {}
        for word in self.word_freqs:
            self.splits[word] = list(word)
        
        # Build initial vocabulary
        self.vocab = set()
        for word in self.word_freqs:
            for char in word:
                self.vocab.add(char)
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Perform merges
        for i in range(num_merges):
            pairs = self.get_stats(self.splits)
            if not pairs:
                print(f"No more pairs to merge at iteration {i}")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.splits = self.merge_symbols(best_pair, self.splits)
            self.merges[best_pair] = i
            merged_token = ''.join(best_pair)
            self.vocab.add(merged_token)
            
            if (i + 1) % 1000 == 0:
                print(f"Merge {i + 1}: {best_pair} -> {merged_token} (freq: {pairs[best_pair]})")
            
            if len(self.vocab) >= self.vocab_size:
                print(f"Reached vocabulary size limit: {len(self.vocab)}")
                break
        
        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[str]:
        """Encode text using learned BPE"""
        words = self.pre_tokenize(text)
        encoded = []
        
        for word in words:
            if word in self.splits:
                encoded.extend(self.splits[word])
            else:
                # Apply learned merges to new words
                word_tokens = list(word)
                
                # Apply merges in order
                for (first, second), merge_order in sorted(self.merges.items(), key=lambda x: x[1]):
                    new_word = []
                    i = 0
                    while i < len(word_tokens):
                        if (i < len(word_tokens) - 1 and 
                            word_tokens[i] == first and 
                            word_tokens[i + 1] == second):
                            new_word.append(first + second)
                            i += 2
                        else:
                            new_word.append(word_tokens[i])
                            i += 1
                    word_tokens = new_word
                
                encoded.extend(word_tokens)
        
        return encoded
    
    def get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a word"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs


class WordPieceTokenizer:
    def __init__(self, vocab_size: int = 32000, unk_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = set()
        self.word_freqs = {}
        self.subword_counts = defaultdict(int)
        
    def pre_tokenize(self, text: str) -> List[str]:
        """Enhanced pre-tokenization for multilingual text"""
        if not text or not text.strip():
            return []
        
        # Enhanced patterns for Indian languages
        patterns = [
            r'[\u0900-\u097F]+',  # Devanagari (Hindi, Marathi)
            r'[a-zA-Z]+',         # English words
            r'\d+',               # Numbers
            r'[^\w\s]'            # Punctuation
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        matches = re.findall(combined_pattern, text)
        
        # Flatten and filter
        words = []
        for match in matches:
            for group in match:
                if group and group.strip():
                    words.append(group.lower())
        
        return words
    
    def get_word_pieces(self, word: str) -> List[str]:
        """Generate all possible subword pieces for a word"""
        pieces = []
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                piece = word[i:j]
                if i > 0:
                    piece = "##" + piece  # WordPiece continuation prefix
                pieces.append(piece)
        return pieces
    
    def calculate_likelihood(self, piece: str) -> float:
        """Calculate likelihood score for WordPiece - optimized version"""
        # Remove ## prefix for counting
        clean_piece = piece[2:] if piece.startswith("##") else piece
        
        # Simple frequency-based scoring - count words containing this piece
        count = 0
        for word, freq in self.word_freqs.items():
            if clean_piece in word:
                count += freq
        
        # Return normalized score by length
        return count / len(clean_piece) if len(clean_piece) > 0 else 0
    
    def train(self, corpus: List[str], vocab_size: int = 32000):
        """Train WordPiece on the corpus using iterative algorithm - optimized"""
        print(f"Training WordPiece with vocab size {vocab_size}...")
        
        # Pre-tokenize and count word frequencies
        all_words = []
        for text in corpus:
            words = self.pre_tokenize(text)
            all_words.extend(words)
        
        self.word_freqs = Counter(all_words)
        print(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize vocabulary with special tokens and characters
        self.vocab = set([self.unk_token])
        for word in self.word_freqs:
            for char in word:
                self.vocab.add(char)
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Generate all possible candidate pieces once
        print("Generating candidate pieces...")
        all_candidates = set()
        for word in self.word_freqs:
            pieces = self.get_word_pieces(word)
            for piece in pieces:
                if piece not in self.vocab and len(piece) > 1:
                    all_candidates.add(piece)
        
        print(f"Generated {len(all_candidates)} candidate pieces")
        
        # Add pieces in batches for efficiency
        iteration = 0
        while len(self.vocab) < vocab_size and all_candidates:
            iteration += 1
            
            # Sample candidates if too many (for efficiency)
            candidates_to_score = all_candidates
            if len(all_candidates) > 1000:
                candidates_to_score = set(list(all_candidates)[:1000])
            
            # Score candidates
            best_piece = None
            best_score = -1
            
            for piece in candidates_to_score:
                if piece not in self.vocab:
                    score = self.calculate_likelihood(piece)
                    if score > best_score:
                        best_score = score
                        best_piece = piece
            
            if best_piece is None:
                break
            
            self.vocab.add(best_piece)
            all_candidates.remove(best_piece)
            
            if len(self.vocab) % 100 == 0:
                print(f"Iteration {iteration}: Vocabulary size: {len(self.vocab)}, Added: {best_piece} (score: {best_score:.4f})")
            
            # Stop if we've reached the target vocab size
            if len(self.vocab) >= vocab_size:
                break
        
        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[str]:
        """Encode text using WordPiece (greedy longest-match)"""
        words = self.pre_tokenize(text)
        tokens = []
        
        for word in words:
            word_tokens = []
            start = 0
            
            while start < len(word):
                end = len(word)
                cur_substr = None
                
                # Find longest matching subword (greedy approach)
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    word_tokens.append(self.unk_token)
                    start += 1
                else:
                    word_tokens.append(cur_substr)
                    if cur_substr.startswith("##"):
                        start = start + len(cur_substr) - 2  # -2 for "##"
                    else:
                        start = start + len(cur_substr)
            
            tokens.extend(word_tokens)
        
        return tokens


def load_assignment1_corpus():
    """Load corpus from Assignment 1 (Hindi and Marathi sentences)"""
    corpus_texts = []
    
    # Try to load Hindi sentences
    hindi_file = os.path.join("..", "Lab1", "tokenized_hindi_sentences.json")
    if os.path.exists(hindi_file):
        try:
            with open(hindi_file, 'r', encoding='utf-8') as f:
                hindi_data = json.load(f)
                # Flatten the nested structure
                for paragraph in hindi_data:
                    for sentence in paragraph:
                        if sentence.strip():
                            corpus_texts.append(sentence)
            print(f"Loaded {len([s for p in hindi_data for s in p])} Hindi sentences")
        except Exception as e:
            print(f"Error loading Hindi data: {e}")
    
    # Try to load Marathi sentences
    marathi_file = os.path.join("..", "Lab1", "tokenized_marathi_sentences.json")
    if os.path.exists(marathi_file):
        try:
            with open(marathi_file, 'r', encoding='utf-8') as f:
                marathi_data = json.load(f)
                # Flatten the nested structure
                for paragraph in marathi_data:
                    for sentence in paragraph:
                        if sentence.strip():
                            corpus_texts.append(sentence)
            print(f"Loaded {len([s for p in marathi_data for s in p])} Marathi sentences")
        except Exception as e:
            print(f"Error loading Marathi data: {e}")
    
    if not corpus_texts:
        print("No corpus found from Assignment 1. Using sample data.")
        corpus_texts = [
            "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर विज्ञान का एक क्षेत्र है",
            "यह भाषा और कंप्यूटर के बीच संपर्क का अध्ययन करता है",
            "मशीन लर्निंग और एआई में इसका महत्वपूर्ण योगदान है",
            "प्राकृतिक भाषा संस्करण हे संगणक विज्ञानाचे एक क्षेत्र आहे",
            "हे भाषा आणि संगणक यांच्यातील संपर्काचा अभ्यास करते"
        ]
    
    print(f"Total corpus size: {len(corpus_texts)} sentences")
    return corpus_texts


def main():
    print("=" * 60)
    print("BPE and WordPiece Implementation for Assignment 1 Corpus")
    print("=" * 60)
    
    # Load corpus from Assignment 1
    corpus = load_assignment1_corpus()
    
    if not corpus:
        print("Failed to load corpus. Exiting.")
        return
    
    # Training parameters as specified
    NUM_MERGES = 32000
    VOCAB_SIZE = 32000
    
    print(f"\nTraining with {NUM_MERGES} merge steps and vocab size {VOCAB_SIZE}")
    print("-" * 60)
    
    # Train BPE Tokenizer
    print("\n1. Training BPE Tokenizer...")
    bpe = BPETokenizer(vocab_size=VOCAB_SIZE)
    bpe.train(corpus, num_merges=NUM_MERGES)
    
    print("\n2. Training WordPiece Tokenizer...")
    wordpiece = WordPieceTokenizer(vocab_size=VOCAB_SIZE)
    wordpiece.train(corpus, vocab_size=VOCAB_SIZE)
    
    # Test encoding with sample texts
    print("\n" + "=" * 60)
    print("TESTING TOKENIZERS")
    print("=" * 60)
    
    test_texts = [
        "प्राकृतिक भाषा प्रसंस्करण",
        "मशीन लर्निंग एल्गोरिदम",
        "संगणक विज्ञान",
        "natural language processing"
    ]
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {test_text}")
        bpe_tokens = bpe.encode(test_text)
        wp_tokens = wordpiece.encode(test_text)
        
        print(f"BPE tokens ({len(bpe_tokens)}): {bpe_tokens}")
        print(f"WordPiece tokens ({len(wp_tokens)}): {wp_tokens}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Corpus size: {len(corpus)} sentences")
    print(f"BPE final vocabulary size: {len(bpe.vocab)}")
    print(f"BPE total merges performed: {len(bpe.merges)}")
    print(f"WordPiece final vocabulary size: {len(wordpiece.vocab)}")
    
    # Show some example vocab items
    print("\nSample BPE vocabulary items:")
    bpe_vocab_list = list(bpe.vocab)
    for token in bpe_vocab_list[:20]:
        print(f"  '{token}'")
    
    print("\nSample WordPiece vocabulary items:")
    wp_vocab_list = list(wordpiece.vocab)
    for token in wp_vocab_list[:20]:
        print(f"  '{token}'")

if __name__ == "__main__":
    main()