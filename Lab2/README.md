# Lab 2: Finite Automata and Morphological Analysis

## Overview
This lab implements two key components of computational linguistics: Deterministic Finite Automata (DFA) for English word validation and Finite State Transducers (FST) for English morphological analysis, particularly focusing on plural formation rules.

## Files
- `dfa1.py` - DFA implementation for English word validation
- `dfa2.py` - Morphological FST for English plural analysis  
- `dfa1_output.txt` - Output from DFA word validation tests
- `dfa2_output.txt` - Output from morphological analysis
- `brown_nouns.txt` - Input corpus of English nouns
- `English_Word_DFA.gv/.png` - DFA visualization
- `EnglishMorphologyFST.gv/.png` - FST visualization

## Part 1: English Word DFA (`dfa1.py`)

### Purpose
Validates English words using a DFA that accepts:
- Words starting with lowercase letters (e.g., "cat", "dog")  
- Words starting with uppercase followed by lowercase (e.g., "Cat", "Dog")

### DFA States
- `q0` - Start state
- `q1` - Lowercase start state (accepting)
- `q2` - Uppercase start state (accepting)  
- `q_reject` - Reject state for invalid patterns

### Validation Rules
**✅ Accepted:**
- `cat`, `dog`, `programming` (all lowercase)
- `Cat`, `Dog`, `Programming` (initial capital)

**❌ Rejected:**
- `dog1`, `1dog` (contains/starts with digits)
- `DogHouse`, `CamelCase` (multiple capitals)
- `dog_house`, `hello world` (special chars/spaces)
- `""` (empty string)

### Features
- Comprehensive test case generation
- Automatic visualization using Graphviz
- Statistical analysis of acceptance/rejection patterns
- Special character handling with proper quoting

## Part 2: Morphological FST (`dfa2.py`)

### Purpose
Analyzes English word morphology, specifically plural formation rules using Finite State Transducers.

### Morphological Rules Implemented

#### 1. Regular Plurals (Add -s)
- `cat` → `cats`, `dog` → `dogs`, `book` → `books`

#### 2. Sibilant Plurals (Add -es) 
- Words ending in `s, x, z, ch, sh`
- `fox` → `foxes`, `watch` → `watches`, `glass` → `glasses`

#### 3. Y-to-IES Transformation
- Consonant + y → ies
- `try` → `tries`, `fly` → `flies`, `baby` → `babies`

#### 4. Irregular Plurals
- `child` → `children`, `foot` → `feet`, `tooth` → `teeth`
- `man` → `men`, `woman` → `women`, `mouse` → `mice`

### Analysis Output Format
```
foxes = fox+N+PL
cats = cat+N+PL  
child = child+N+SG
children = child+N+PL
```

### FST Components
- **States**: START, ROOT, various ending states, ACCEPT, REJECT
- **Transitions**: Morphological rule applications
- **Output**: Morphological analysis with features (+N+PL, +N+SG)

## Visualization

### DFA Visualization
- States represented as circles
- Accepting states as double circles  
- Transitions labeled with input symbols
- Generated using Graphviz in multiple formats

### FST Visualization  
- Multiple diagram styles (detailed, simplified, DFA-style)
- Rule-based transitions showing morphological transformations
- Clear representation of accept/reject paths

## Usage

### Running DFA Validation
```bash
python dfa1.py
```
- Tests predefined word list
- Generates additional test cases
- Creates visualization
- Outputs results to `dfa1_output.txt`

### Running Morphological Analysis
```bash  
python dfa2.py
```
- Processes `brown_nouns.txt` corpus
- Analyzes each word for morphological structure
- Generates FST visualizations
- Outputs analyses to `dfa2_output.txt`

## Key Features

### Error Handling
- Graceful handling of malformed input
- Fallback for missing corpus files
- Comprehensive edge case testing

### Performance Optimizations
- Efficient state transition implementations
- Memory-conscious corpus processing
- Optimized regex patterns

### Educational Value
- Clear separation of DFA and FST concepts
- Comprehensive test coverage
- Multiple visualization styles
- Detailed morphological rule explanations

## Applications
- Morphological analysis preprocessing
- Word validation in spell checkers
- Educational tools for finite automata theory
- Linguistic research on English morphology
- Foundation for more complex NLP pipelines

## Requirements
- `automathon` (DFA implementation)
- `graphviz` (visualization)
- `re` (regex processing)
- `string` (text utilities)

## Technical Notes
- Uses Unicode-safe string processing
- Implements proper Graphviz node quoting
- Supports multiple output formats (PNG, PDF, SVG)
- Memory-efficient corpus processing