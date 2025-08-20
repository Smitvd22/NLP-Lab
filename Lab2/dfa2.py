class MorphologicalFST:
    def __init__(self):
        self.states = {
            'START',
            'ROOT',
            'S_END',
            'Z_END',
            'X_END',
            'CH_END',
            'SH_END',
            'Y_END',
            'ACCEPT',
            'REJECT'
        }

        self.irregular_patterns = {
            'children': 'child',
            'feet': 'foot',
            'teeth': 'tooth',
            'men': 'man',
            'women': 'woman',
            'mice': 'mouse',
            'geese': 'goose',
            'people': 'person'
        }

        self.non_plural_es_words = {
            'analyses', 'bases', 'crises', 'diagnoses', 'hypotheses',
            'oases', 'parentheses', 'syntheses', 'theses'
        }

    def analyze_word(self, word):
        if not word or not word.isalpha():
            return f"{word}: Invalid Word"

        word = word.lower().strip()

        if word in self.irregular_patterns:
            root = self.irregular_patterns[word]
            return f"{word} = {root}+N+PL"
        
        analysis = self.analyze_plural_morphology(word)

        if analysis:
            return analysis
        else:
            return f"{word} = {word}+N+SG"
        
    def analyze_plural_morphology(self, word):
        if word.endswith('es') and len(word) > 2:
            if word in self.non_plural_es_words:
                return None
            
            root_candidate = word[:-2] 

            if (root_candidate.endswith(('s', 'x', 'z')) or root_candidate.endswith(('ch', 'sh'))):
                return f"{word} = {root_candidate}+N+PL"
            
        if word.endswith('ies') and len(word)>3:
            root_candidate = word[:-3] + 'y'
            if not word.endswith('eies'):
                return f"{word} = {root_candidate}+N+PL"
            
        if word.endswith('s') and len(word) > 1:
            if not word.endswith(('es', 'ies')):
                root_candidate = word[:-1]

                if not self.is_naturally_s_ending(root_candidate, word):
                    return f"{word} = {root_candidate}+N+PL"
                
        return None
    
    def is_naturally_s_ending(self, root, word):
        naturally_s_ending = [
            'lens', 'bus', 'gas', 'glass', 'class', 'mass', 'pass', 'bass',
            'grass', 'dress', 'stress', 'press', 'chess', 'mess', 'less',
            'business', 'princess', 'process', 'success', 'access', 'address'
        ]

        if word in naturally_s_ending:
            return True
        
        if len(root) < 2:
            return True

        return False
    
    def process_corpus(self, filename):
        analyses = []
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    words = line.strip().split()
                    for word in words:
                        clean_word = ''.join(c for c in word if c.isalpha())
                        if clean_word:
                            analysis = self.analyze_word(clean_word)
                            analyses.append(analysis)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return []
        
        return analyses
    
def main():
    fst = MorphologicalFST()
    test_words = [
        # Rule 1: E insertion
        'foxes',      # fox + es
        'watches',    # watch + es  
        'boxes',      # box + es
        'glasses',    # glass + es
        'wishes',     # wish + es
        
        # Rule 2: Y replacement
        'tries',      # try -> tries
        'flies',      # fly -> flies
        'babies',     # baby -> babies
        'cities',     # city -> cities
        
        # Rule 3: S addition
        'bags',       # bag + s
        'cats',       # cat + s
        'dogs',       # dog + s
        'books',      # book + s
        
        # Singular words
        'investigation',
        'primary',
        'election',
        'evidence',
        'jury',
        'manner',
        
        # Edge cases
        'fox',        # Should be singular
        'lens',       # Naturally ends in s
        'bus',        # Naturally ends in s
        
        # Invalid cases
        'foxs',       # Invalid plural
    ]

    for word in test_words:
        analysis=fst.analyze_word(word)
        print(analysis)

    analyses = fst.process_corpus("Lab2/brown_nouns.txt")

    with open("lab2/output.txt", "w", encoding='utf-8') as f:
        for a in analyses:
            f.write(a+"\n")

if __name__ == "__main__":
    main()