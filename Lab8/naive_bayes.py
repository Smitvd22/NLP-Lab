import re
import collections
import math

# ==========================================
# 1. DATASET & PREPROCESSING
# ==========================================

raw_data = [
    ("Check out https://example.com for more info!", "Inform"),
    ("Order 3 items, get 1 free! Limited offer!!!", "Promo"),
    ("Your package #12345 will arrive tomorrow.", "Inform"),
    ("Win $1000 now, visit http://winbig.com!!!", "Promo"),
    ("Meeting at 3pm, don't forget to bring the files.", "Reminder"),
    ("Exclusive deal for you: buy 2, get 1 free!!!", "Promo"),
    ("Download the report from https://reports.com.", "Inform"),
    ("The meeting is starting in 10 minutes.", "Reminder"),
    ("Reminder: submit your timesheet by 5pm today.", "Reminder")
]

def tokenizer(text):
    """
    Regex-based tokenizer (same as Assignment 1).
    Handles URLs, Numbers, Punctuation, and Words.
    """
    # Patterns from Assignment 1
    url_pattern = r'https?://\S+|www\.\S+'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'
    number_pattern = r'\d+(?:\.\d+)?'
    word_pattern = r"\w+"
    punct_pattern = r'[^\w\s]'

    master_pattern = f'({url_pattern})|({email_pattern})|({date_pattern})|({number_pattern})|({word_pattern})|({punct_pattern})'
    
    tokens = []
    for match in re.finditer(master_pattern, text):
        tokens.append(match.group())
    return tokens

def get_bigrams(tokens):
    # Convert to lower case for generalization
    lower_tokens = [t.lower() for t in tokens]
    if len(lower_tokens) < 2: return []
    return [tuple(lower_tokens[i:i+2]) for i in range(len(lower_tokens)-1)]

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================

def extract_features(sentence):
    """
    Extracts specific features and bigrams.
    """
    tokens = tokenizer(sentence)
    
    # 1. Specific Features (Binary)
    has_url = 1 if re.search(r'https?://|www\.', sentence) else 0
    has_number = 1 if re.search(r'\d', sentence) else 0
    # Checking for '!' specifically as it's common in Promo
    has_exclaim = 1 if '!' in sentence else 0 
    
    # 2. Bigrams
    bigrams = get_bigrams(tokens)
    
    return {
        "has_url": has_url,
        "has_number": has_number,
        "has_exclaim": has_exclaim,
        "bigrams": bigrams
    }

# ==========================================
# 3. NAIVE BAYES TRAINING
# ==========================================

class NaiveBayesClassifier:
    def __init__(self, k=0.3):
        self.k = k
        self.classes = set()
        self.class_counts = collections.defaultdict(int)
        
        # Feature Counts per Class
        self.feature_counts = {
            "has_url": collections.defaultdict(int),
            "has_number": collections.defaultdict(int),
            "has_exclaim": collections.defaultdict(int),
            "bigrams": collections.defaultdict(lambda: collections.defaultdict(int))
        }
        
        # Totals for denominators
        self.total_docs_in_class = collections.defaultdict(int)
        self.total_bigrams_in_class = collections.defaultdict(int)
        self.bigram_vocab = set()

    def train(self, data):
        total_docs = len(data)
        
        # 1. Collect Counts
        for sentence, label in data:
            self.classes.add(label)
            self.class_counts[label] += 1
            self.total_docs_in_class[label] += 1
            
            feats = extract_features(sentence)
            
            # Count Specific Features (Bernoulli: Count docs that have feature)
            if feats["has_url"]: self.feature_counts["has_url"][label] += 1
            if feats["has_number"]: self.feature_counts["has_number"][label] += 1
            if feats["has_exclaim"]: self.feature_counts["has_exclaim"][label] += 1
            
            # Count Bigrams (Multinomial: Count occurrences)
            for bg in feats["bigrams"]:
                self.feature_counts["bigrams"][bg][label] += 1
                self.total_bigrams_in_class[label] += 1
                self.bigram_vocab.add(bg)

        # 2. Calculate Priors: P(Class)
        self.priors = {c: count/total_docs for c, count in self.class_counts.items()}
        
    def get_feature_prob(self, feature_name, class_label, val=1):
        # For binary features: P(F=1 | C) using Laplace smoothing (k=1 standard, but using k=0.3 for consistency)
        # P = (Count(Docs_with_F=1, C) + k) / (Count(Docs_in_C) + k*2)
        count_f = self.feature_counts[feature_name][class_label]
        total_c = self.total_docs_in_class[class_label]
        
        # If we are checking for presence (val=1)
        prob_present = (count_f + self.k) / (total_c + (self.k * 2))
        
        return prob_present if val == 1 else (1 - prob_present)

    def get_bigram_prob(self, bigram, class_label):
        # Multinomial with Add-K smoothing
        # P = (Count(bigram, C) + k) / (Total_Bigrams_in_C + k * V)
        count_w = self.feature_counts["bigrams"][bigram][class_label]
        total_w = self.total_bigrams_in_class[class_label]
        vocab_size = len(self.bigram_vocab)
        
        return (count_w + self.k) / (total_w + (self.k * vocab_size))

    def predict(self, sentence):
        feats = extract_features(sentence)
        scores = {}
        
        print(f"\n--- Predicting for: '{sentence}' ---")
        
        for c in self.classes:
            # Start with Log Prior
            log_prob = math.log(self.priors[c])
            details = [f"Prior({c}): {self.priors[c]:.4f}"]
            
            # 1. Add Log Prob of Specific Features
            # Note: We multiply probabilities for presence AND absence
            for fname in ["has_url", "has_number", "has_exclaim"]:
                val = feats[fname] # 1 or 0
                p = self.get_feature_prob(fname, c, val)
                log_prob += math.log(p)
                details.append(f"P({fname}={val}|{c}): {p:.4f}")
            
            # 2. Add Log Prob of Bigrams
            # We only calculate prob for bigrams present in the sentence
            for bg in feats["bigrams"]:
                # Even if bigram is unseen in training, smoothing handles it
                p = self.get_bigram_prob(bg, c)
                log_prob += math.log(p)
                details.append(f"P({bg}|{c}): {p:.4f}")
                
            scores[c] = log_prob
            print(f"\nClass: {c} (Log Score: {log_prob:.4f})")
            # print(", ".join(details)) # Uncomment to see all probability steps
            
        # Find max score
        predicted_class = max(scores, key=scores.get)
        return predicted_class

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("--- Naive Bayes Classifier Construction ---")
    
    # 1. Preprocessing Output
    print("\n[Step 1] Preprocessed Sentences:")
    for s, l in raw_data:
        toks = tokenizer(s)
        print(f"{l:<10} | {toks}")

    # 2. Train
    nb = NaiveBayesClassifier(k=0.3)
    nb.train(raw_data)
    
    # 3. Show Feature Probabilities (Subset)
    print("\n[Step 2] Feature Probabilities (Examples):")
    for c in nb.classes:
        print(f"\nClass: {c}")
        print(f"  P(has_url=1 | {c})    = {nb.get_feature_prob('has_url', c):.4f}")
        print(f"  P(has_number=1 | {c}) = {nb.get_feature_prob('has_number', c):.4f}")
        print(f"  P(has_exclaim=1 | {c})= {nb.get_feature_prob('has_exclaim', c):.4f}")
    
    # 4. Predict
    test_sentence = "You will get an exclusive offer in the meeting!"
    prediction = nb.predict(test_sentence)
    
    print(f"\n>>> FINAL PREDICTION: {prediction} <<<")