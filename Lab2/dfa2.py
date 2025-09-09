import graphviz

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
    
    def create_fst_diagram(self):
        """Create a graphviz diagram representing the morphological FST"""
        dot = graphviz.Digraph(comment='Morphological FST for English Plurals')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='circle')
        
        # Add states
        dot.node('START', 'START', shape='circle')
        dot.node('ROOT', 'ROOT', shape='circle')
        dot.node('S_END', 'S_END', shape='circle')
        dot.node('Z_END', 'Z_END', shape='circle')
        dot.node('X_END', 'X_END', shape='circle')
        dot.node('CH_END', 'CH_END', shape='circle')
        dot.node('SH_END', 'SH_END', shape='circle')
        dot.node('Y_END', 'Y_END', shape='circle')
        dot.node('ACCEPT', 'ACCEPT', shape='doublecircle')
        dot.node('REJECT', 'REJECT', shape='circle')
        
        # Add transitions representing morphological rules
        
        # From START to ROOT (any alphabetic input)
        dot.edge('START', 'ROOT', 'α/α\n(any letter)')
        
        # From ROOT to various ending states based on morphological patterns
        dot.edge('ROOT', 'S_END', 's/+PL\n(ends with s)')
        dot.edge('ROOT', 'Z_END', 'z/+PL\n(ends with z)')
        dot.edge('ROOT', 'X_END', 'x/+PL\n(ends with x)')
        dot.edge('ROOT', 'CH_END', 'ch/+PL\n(ends with ch)')
        dot.edge('ROOT', 'SH_END', 'sh/+PL\n(ends with sh)')
        dot.edge('ROOT', 'Y_END', 'y/ies\n(y→ies rule)')
        
        # From ending states to ACCEPT
        dot.edge('S_END', 'ACCEPT', 'es/ε\n(add es)')
        dot.edge('Z_END', 'ACCEPT', 'es/ε\n(add es)')
        dot.edge('X_END', 'ACCEPT', 'es/ε\n(add es)')
        dot.edge('CH_END', 'ACCEPT', 'es/ε\n(add es)')
        dot.edge('SH_END', 'ACCEPT', 'es/ε\n(add es)')
        dot.edge('Y_END', 'ACCEPT', 'ies/ε\n(y→ies)')
        
        # Direct path for regular plurals
        dot.edge('ROOT', 'ACCEPT', 's/+PL\n(regular plural)')
        
        # Irregular forms (special transitions)
        dot.edge('START', 'ACCEPT', 'irregular/+PL\n(children, feet, etc.)')
        
        # Singular forms
        dot.edge('ROOT', 'ACCEPT', 'ε/+SG\n(singular)')
        
        # Error transitions to REJECT
        dot.edge('START', 'REJECT', 'invalid/ε\n(invalid input)')
        dot.edge('ROOT', 'REJECT', 'invalid/ε\n(invalid pattern)')
        
        return dot
    
    def create_simplified_fst_diagram(self):
        """Create a simplified FST diagram focusing on main morphological rules"""
        dot = graphviz.Digraph(comment='Simplified Morphological FST')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='circle')
        
        # States
        dot.node('q0', 'START', shape='circle')
        dot.node('q1', 'WORD', shape='circle')
        dot.node('q2', 'S_STEM', shape='circle')
        dot.node('q3', 'Y_STEM', shape='circle')
        dot.node('q4', 'IRREGULAR', shape='circle')
        dot.node('q5', 'ACCEPT', shape='doublecircle')
        
        # Main transitions
        dot.edge('q0', 'q1', 'letter/letter')
        dot.edge('q1', 'q1', 'letter/letter')
        
        # Plural formation rules
        dot.edge('q1', 'q2', 's,x,z,ch,sh/stem')
        dot.edge('q2', 'q5', 'es/+N+PL')
        
        dot.edge('q1', 'q3', 'consonant+y/stem')
        dot.edge('q3', 'q5', 'ies/+N+PL')
        
        dot.edge('q1', 'q5', 's/+N+PL\n(regular)')
        dot.edge('q1', 'q5', 'ε/+N+SG\n(singular)')
        
        # Irregular plurals
        dot.edge('q0', 'q4', 'irregular_word/stem')
        dot.edge('q4', 'q5', 'ε/+N+PL')
        
        return dot
    
    def create_english_morphology_fst(self):
        """Create FST diagram for English morphological analysis similar to DFA1 style"""
        dot = graphviz.Digraph(comment='English Morphological FST')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='circle', fontsize='10')
        dot.attr('edge', fontsize='9')
        
        # Define states matching the original FST
        dot.node('q0', 'START', shape='circle', style='bold')
        dot.node('q1', 'READING', shape='circle')
        dot.node('q2', 'S_ENDING', shape='circle')
        dot.node('q3', 'ES_NEEDED', shape='circle')
        dot.node('q4', 'Y_ENDING', shape='circle')
        dot.node('q5', 'IRREGULAR', shape='circle')
        dot.node('q6', 'ACCEPT', shape='doublecircle', style='bold')
        dot.node('q7', 'REJECT', shape='circle', style='filled', fillcolor='lightgray')
        
        # Transitions for word reading
        dot.edge('q0', 'q1', 'α (letter)')
        dot.edge('q1', 'q1', 'α (continue reading)')
        
        # Regular plural with 's'
        dot.edge('q1', 'q2', 'word ending')
        dot.edge('q2', 'q6', 's → +N+PL\\n(cats, dogs)')
        
        # Plural requiring 'es' (s, x, z, ch, sh endings)
        dot.edge('q1', 'q3', 's|x|z|ch|sh ending')
        dot.edge('q3', 'q6', 'es → +N+PL\\n(foxes, watches)')
        
        # Y to ies transformation
        dot.edge('q1', 'q4', 'consonant+y ending')
        dot.edge('q4', 'q6', 'y→ies → +N+PL\\n(tries, flies)')
        
        # Irregular plurals
        dot.edge('q0', 'q5', 'irregular pattern')
        dot.edge('q5', 'q6', 'children→child+N+PL\\nfeet→foot+N+PL')
        
        # Singular words
        dot.edge('q2', 'q6', 'no suffix → +N+SG\\n(cat, book)')
        
        # Error handling
        dot.edge('q0', 'q7', 'invalid input')
        dot.edge('q1', 'q7', 'invalid pattern')
        
        return dot
    
    def generate_visualization(self, filename_prefix="MorphologicalFST"):
        """Generate both detailed and simplified FST visualizations"""
        
        # Generate detailed FST
        detailed_fst = self.create_fst_diagram()
        detailed_fst.render(f'{filename_prefix}_detailed', format='png', cleanup=True)
        
        # Generate simplified FST
        simplified_fst = self.create_simplified_fst_diagram()
        simplified_fst.render(f'{filename_prefix}_simplified', format='png', cleanup=True)
        
        print(f"FST diagrams generated:")
        print(f"- {filename_prefix}_detailed.png")
        print(f"- {filename_prefix}_simplified.png")
    
    def view(self, filename="EnglishMorphologyFST"):
        """Generate visualization similar to DFA1's view method"""
        dot = self.create_english_morphology_fst()
        
        # Save the .gv file
        dot.save(f'{filename}.gv')
        
        # Render to PNG
        dot.render(filename, format='png', cleanup=True)
        
        print(f"FST visualization saved as {filename}.gv and {filename}.png")
    
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

    print("English Morphological FST Analysis Examples:")
    print("=" * 50)
    
    for word in test_words:
        analysis = fst.analyze_word(word)
        print(analysis)

    print("\n" + "=" * 50)
    print("Processing corpus and generating FST visualizations...")
    
    # Process a corpus file (brown_nouns.txt) and write analyses to Lab2/output.txt
    corpus_file = "Lab2/brown_nouns.txt"
    analyses = fst.process_corpus(corpus_file)

    outpath = "Lab2/dfa2_output.txt"
    try:
        with open(outpath, 'w', encoding='utf-8') as out_file:
            for analysis in analyses:
                out_file.write(f"{analysis}\n")
        print(f"Corpus processing complete. Output saved to {outpath}")
    except Exception as e:
        print(f"Failed to write output file '{outpath}': {e}")
    
    # Generate FST visualizations
    print("\nGenerating FST visualizations...")
    
    # Generate detailed and simplified visualizations
    fst.generate_visualization("Lab2/MorphologicalFST")
    
    # Generate DFA-style visualization
    fst.view("Lab2/EnglishMorphologyFST")
    
    print("\nAll FST visualizations generated successfully!")

if __name__ == "__main__":
    main()