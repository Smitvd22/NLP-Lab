import collections
import math
import sys

# ==========================================
# 1. DATA LOADING & PREPROCESSING 
# ==========================================

def load_data(filename):
    sentences = []
    current_sentence = []
    
    print(f"Detecting format for {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Heuristic to check format
        sample_line = ""
        for l in lines[:10]:
            if l.strip():
                sample_line = l.strip()
                break
        
        if not sample_line:
            print("Error: File appears empty.")
            return []

        # FORMAT 1: Underscore-Tags (The_DT dollar_NN)
        # We check for underscore and spaces on the same line
        if "_" in sample_line and len(sample_line.split()) > 1 and "/" not in sample_line:
            print("Format Detected: Underscore-Tags (Word_Tag)")
            for line in lines:
                if not line.strip(): continue
                pairs = line.strip().split()
                sent = []
                for p in pairs:
                    # Handle Word_Tag (e.g. dollar_NN)
                    if '_' in p:
                        try:
                            # Split from right to handle words that might have underscores
                            word, tag = p.rsplit('_', 1)
                            sent.append((word, tag))
                        except ValueError:
                            continue
                if sent:
                    sentences.append(sent)

        # FORMAT 2: Standard Slash-Tags (The/DT dollar/NN)
        elif "/" in sample_line and len(sample_line.split()) > 1:
            print("Format Detected: Standard Slash-Tags (Word/Tag)")
            for line in lines:
                if not line.strip(): continue
                pairs = line.strip().split()
                sent = []
                for p in pairs:
                    if '/' in p:
                        try:
                            word, tag = p.rsplit('/', 1)
                            sent.append((word, tag))
                        except ValueError:
                            continue
                if sent:
                    sentences.append(sent)

        # FORMAT 3: Column Format / CoNLL (One word per line)
        else:
            print("Format Detected: Column/Vertical Format (Word Tag)")
            for line in lines:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[-1] 
                    current_sentence.append((word, tag))
            
            if current_sentence:
                sentences.append(current_sentence)

    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        exit()
        
    return sentences

def k_fold_split(data, k=3):
    """Splits data into k folds."""
    if not data:
        return []
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(data)
        test = data[start:end]
        train = data[:start] + data[end:]
        folds.append((train, test))
    return folds

# ==========================================
# 2. HIDDEN MARKOV MODEL (HMM)
# ==========================================

class HMMTagger:
    def __init__(self):
        self.transitions = collections.defaultdict(int)
        self.emissions = collections.defaultdict(int)
        self.tag_counts = collections.defaultdict(int)
        self.vocab = set()
        self.tags = set()
        
    def train(self, sentences):
        if not sentences: return

        for sent in sentences:
            prev_tag = "<START>"
            self.tags.add(prev_tag)
            
            for word, tag in sent:
                self.vocab.add(word)
                self.tags.add(tag)
                
                self.transitions[(prev_tag, tag)] += 1
                self.emissions[(tag, word)] += 1
                self.tag_counts[tag] += 1
                self.tag_counts[prev_tag] += 1
                
                prev_tag = tag
            
            self.transitions[(prev_tag, "<END>")] += 1
            self.tag_counts[prev_tag] += 1
            self.tags.add("<END>")

    def get_transition_prob(self, prev_tag, curr_tag):
        count = self.transitions[(prev_tag, curr_tag)]
        total = self.tag_counts[prev_tag]
        vocab_size = len(self.tags)
        # Add-1 Smoothing
        return (count + 1) / (total + vocab_size)

    def get_emission_prob(self, tag, word):
        count = self.emissions[(tag, word)]
        total = self.tag_counts[tag]
        vocab_size = len(self.vocab)
        # Add-1 Smoothing handles unknown words naturally
        # If word is unknown, count is 0, returns 1/(total+V)
        return (count + 1) / (total + vocab_size)

# ==========================================
# 3. VITERBI DECODING
# ==========================================

def viterbi(model, sentence_words):
    if not sentence_words: return []
    
    N = len(sentence_words)
    states = list(model.tags)
    if "<START>" in states: states.remove("<START>")
    if "<END>" in states: states.remove("<END>")
    
    if not states: return [] # Handle empty model case

    # Use lists for speed
    v_score = [{} for _ in range(N)]
    backpointer = [{} for _ in range(N)]
    
    # Initialization
    first_word = sentence_words[0]
    for tag in states:
        trans = math.log(model.get_transition_prob("<START>", tag))
        emit = math.log(model.get_emission_prob(tag, first_word))
        v_score[0][tag] = trans + emit
        backpointer[0][tag] = "<START>"

    # Recursion
    for t in range(1, N):
        word = sentence_words[t]
        for curr_tag in states:
            emit_prob = math.log(model.get_emission_prob(curr_tag, word))
            best_score = -float('inf')
            best_prev = None
            
            for prev_tag in states:
                if prev_tag in v_score[t-1]:
                    trans_prob = math.log(model.get_transition_prob(prev_tag, curr_tag))
                    score = v_score[t-1][prev_tag] + trans_prob + emit_prob
                    
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag
            
            if best_score != -float('inf'):
                v_score[t][curr_tag] = best_score
                backpointer[t][curr_tag] = best_prev

    # Termination
    best_final_score = -float('inf')
    best_last_tag = states[0]
    for tag in states:
        if tag in v_score[N-1]:
            if v_score[N-1][tag] > best_final_score:
                best_final_score = v_score[N-1][tag]
                best_last_tag = tag

    # Backtrack
    best_path = [best_last_tag]
    current = best_last_tag
    for t in range(N-1, 0, -1):
        if current in backpointer[t] and backpointer[t][current]:
            prev = backpointer[t][current]
            best_path.insert(0, prev)
            current = prev
        else:
            best_path.insert(0, states[0])
            
    return best_path

# ==========================================
# 4. EVALUATION
# ==========================================

def evaluate(true_seqs, pred_seqs):
    true_flat = [t for s in true_seqs for t in s]
    pred_flat = [t for s in pred_seqs for t in s]
    
    if not true_flat:
        return 0.0, 0.0, 0.0, 0.0
    
    correct = sum(1 for t, p in zip(true_flat, pred_flat) if t == p)
    accuracy = correct / len(true_flat)
    
    # Macro-F1
    tags = set(true_flat)
    prec_sum, rec_sum, f1_sum = 0, 0, 0
    n_tags = 0
    
    for t in tags:
        tp = sum(1 for i in range(len(true_flat)) if true_flat[i]==t and pred_flat[i]==t)
        fp = sum(1 for i in range(len(true_flat)) if true_flat[i]!=t and pred_flat[i]==t)
        fn = sum(1 for i in range(len(true_flat)) if true_flat[i]==t and pred_flat[i]!=t)
        
        p = tp / (tp + fp) if (tp+fp) > 0 else 0
        r = tp / (tp + fn) if (tp+fn) > 0 else 0
        f = 2*p*r / (p+r) if (p+r) > 0 else 0
        
        # Only include tags that actually appear (to avoid div by zero)
        if (tp + fn) > 0 or (tp + fp) > 0:
            prec_sum += p
            rec_sum += r
            f1_sum += f
            n_tags += 1
    
    if n_tags == 0: return 0.0, 0.0, 0.0, accuracy

    return prec_sum/n_tags, rec_sum/n_tags, f1_sum/n_tags, accuracy

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    filename = "lab10/wsj_pos_tagged_en.txt" 
    
    print("--- Loading Data ---")
    data = load_data(filename)
    print(f"Successfully Loaded: {len(data)} sentences.")
    
    if len(data) < 5:
        print("\nCRITICAL WARNING: Very few sentences loaded.")
        print("Please check the file path or format.")
        if len(data) > 0:
            print(f"Sample sentence 1: {data[0]}")
        exit()

    k = 3
    folds = k_fold_split(data, k)
    
    t_prec, t_rec, t_f1 = 0, 0, 0
    
    for i, (train, test) in enumerate(folds):
        print(f"\n--- Fold {i+1}/{k} ---")
        print(f"Train: {len(train)} | Test: {len(test)}")
        
        if len(train) == 0 or len(test) == 0:
            print("Skipping fold due to empty data.")
            continue
            
        hmm = HMMTagger()
        hmm.train(train)
        
        trues, preds = [], []
        print(f"Decoding {len(test)} test sentences...")
        
        for idx, sent in enumerate(test):
            if not sent: continue
            w = [x[0] for x in sent]
            t = [x[1] for x in sent]
            p = viterbi(hmm, w)
            trues.append(t)
            preds.append(p)
            
            if idx % 500 == 0 and idx > 0:
                print(f"Processed {idx}...")
                
        p, r, f, a = evaluate(trues, preds)
        print(f"Fold Results > Prec: {p:.4f} | Rec: {r:.4f} | F1: {f:.4f} | Acc: {a:.4f}")
        t_prec += p; t_rec += r; t_f1 += f

    print(f"Avg Precision: {t_prec/k:.4f}")
    print(f"Avg Recall:    {t_rec/k:.4f}")
    print(f"Avg F1 Score:  {t_f1/k:.4f}")