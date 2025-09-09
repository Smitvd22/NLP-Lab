# Lab6 — N-gram Smoothing and Sentence Generation

This folder contains a compact, runnable implementation for:

- Katz Backoff smoothing (`katz_backoff.py`) for n-gram orders up to 4
- Interpolated Kneser-Ney smoothing (`kneser_ney.py`) for n-gram orders up to 4
- Sentence generation (greedy and beam search) (`sentence_generation.py`)

Results and generated sentences are written under `Lab6/results/`.

Usage
-----

1. Ensure Lab4 data files are present in the parent `Lab4/` folder:
   - `hindi_unigrams.tsv`, `hindi_bigrams.tsv`, `hindi_trigrams.tsv`, `hindi_quadragrams.tsv`
   - same for `marathi_*` files

2. From repository root run a small script to train and generate sentences. Example:

```python
from Lab6.katz_backoff import KatzBackoffModel
from Lab6.kneser_ney import KneserNeyModel
from Lab6.sentence_generation import generate_sentences_for_model

# Katz Backoff
katz = KatzBackoffModel(max_n=4)
katz.train('hindi')
generate_sentences_for_model(katz, 'Katz Backoff', 'hindi', num_sentences=100)

# Kneser-Ney
kn = KneserNeyModel(max_n=4)
kn.train('hindi')
generate_sentences_for_model(kn, 'Kneser-Ney', 'hindi', num_sentences=100)
```

Outputs
-------

- `Lab6/results/{model_name}_{language}_generated_sentences.json` — JSON with greedy and beam outputs
- `Lab6/results/{model_name}_{language}_greedy_sentences.txt` — human readable greedy sentences
- `Lab6/results/{model_name}_{language}_beam_search_sentences.txt` — human readable beam sentences

Notes
-----

- The implementations here are written to be clear and robust for the Lab environment. They use the precomputed TSV files from Lab4.
- If you want to run all combinations and both languages in one go, create a small driver script (example above).
# Lab 6: Advanced N-gram Language Models

This lab implements advanced smoothing techniques for n-gram language models and sentence generation algorithms.

## Overview

This implementation includes:

1. **Katz Backoff Smoothing** for quadrigram model
2. **Kneser-Ney Smoothing** for quadrigram model  
3. **Sentence Generation** using:
   - Greedy Approach (Maximum Likelihood Estimation)
   - Beam Search with beam size=20

## Files Description

### Core Implementation Files

- `katz_backoff.py` - Implementation of Katz Backoff smoothing algorithm
- `kneser_ney.py` - Implementation of Kneser-Ney smoothing algorithm
- `sentence_generation.py` - Sentence generation using greedy and beam search
- `main.py` - Main script that runs all experiments
- `requirements.txt` - Required Python packages

### Generated Output Files

- `katz_backoff_*_model.json` - Saved Katz Backoff models
- `kneser_ney_*_model.json` - Saved Kneser-Ney models
- `*_generated_sentences.json` - Generated sentences with statistics
- `*_greedy_sentences.txt` - Greedy generated sentences (readable format)
- `*_beam_search_sentences.txt` - Beam search sentences (readable format)
- `comparison_report.txt` - Comprehensive comparison report

Quick run examples (PowerShell)
------------------------------

Run Katz Backoff (Hindi) and generate 100 sentences:

```powershell
python -c "from Lab6.katz_backoff import main; main(['hindi'], num_sentences=100)"
```

Run Kneser-Ney (Hindi) and generate 100 sentences:

```powershell
python -c "from Lab6.kneser_ney import main; main(['hindi'], num_sentences=100)"
```

Quick smoke test (2 sentences) for both models:

```powershell
python -c "from Lab6.katz_backoff import main; main(['hindi'], num_sentences=2)"
python -c "from Lab6.kneser_ney import main; main(['hindi'], num_sentences=2)"
```

## Algorithm Details

### 1. Katz Backoff Smoothing

Katz backoff uses Good-Turing discounting for observed n-grams and backs off to lower-order n-grams for unseen n-grams:

- **Good-Turing Discounting**: For count c, use (c+1) × N(c+1) / N(c)
- **Backoff Weights**: Computed to ensure probability distribution sums to 1
- **Recursive Backoff**: If n-gram unseen, back off to (n-1)-gram with appropriate weight

### 2. Kneser-Ney Smoothing

Kneser-Ney uses absolute discounting with continuation counts:

- **Absolute Discounting**: Subtract fixed discount D from all counts
- **Continuation Counts**: Count unique contexts where word appears
- **Interpolation**: Combine discounted probability with lower-order continuation probability

### 3. Sentence Generation

#### Greedy Approach
- At each step, choose word with highest conditional probability
- Fast but may get stuck in repetitive patterns
- Uses maximum likelihood estimation

#### Beam Search
- Maintains beam of top-k partial sentences
- Explores multiple paths simultaneously
- Better diversity and quality than greedy
- Beam size = 20 for good balance of quality and speed

## Usage

### Running the Complete Experiment

```bash
cd Lab6
python main.py
```

This will:
1. Train both models on Hindi and Marathi data
2. Generate 100 sentences using each model and approach
3. Save all results and generate comparison report

### Running Individual Components

```python
# Katz Backoff Model
from katz_backoff import KatzBackoffModel
model = KatzBackoffModel(max_n=4)
model.train('hindi')
prob = model.get_probability(('यह', 'एक', 'अच्छा', 'दिन'))

# Kneser-Ney Model  
from kneser_ney import KneserNeyModel
model = KneserNeyModel(max_n=4)
model.train('hindi')
prob = model.get_probability(('यह', 'एक', 'अच्छा', 'दिन'))

# Sentence Generation
from sentence_generation import SentenceGenerator
generator = SentenceGenerator(model)
sentences = generator.greedy_generation(10)
beam_sentences = generator.beam_search_generation(10, beam_size=20)
```

## Input Data

The implementation uses pre-computed n-gram data from Lab 4:
- `hindi_unigrams.tsv`, `hindi_bigrams.tsv`, `hindi_trigrams.tsv`, `hindi_quadragrams.tsv`
- `marathi_unigrams.tsv`, `marathi_bigrams.tsv`, `marathi_trigrams.tsv`, `marathi_quadragrams.tsv`

## Output Analysis

### Generated Files

For each language and model combination:
- JSON files with complete results and statistics
- Text files with human-readable generated sentences
- Comparison report with side-by-side analysis

### Evaluation Metrics

- **Average sentence length**
- **Length range** (min-max words)
- **Type-token ratio** (vocabulary diversity)
- **Unique words count**
- **Training time**

### Sample Output Structure

```
Lab6/
├── katz_backoff_hindi_model.json
├── katz_backoff_hindi_generated_sentences.json
├── katz_backoff_hindi_greedy_sentences.txt
├── katz_backoff_hindi_beam_search_sentences.txt
├── kneser_ney_hindi_model.json
├── kneser_ney_hindi_generated_sentences.json
├── kneser_ney_hindi_greedy_sentences.txt
├── kneser_ney_hindi_beam_search_sentences.txt
├── [similar files for marathi]
└── comparison_report.txt
```

## Technical Implementation Notes

### Memory Optimization
- Uses defaultdict and efficient data structures
- Streams data processing where possible
- Configurable vocabulary size limits

### Robustness Features
- Handles unseen n-grams gracefully
- Fallback probabilities for edge cases
- Error handling and logging
- Configurable parameters

### Performance
- Vectorized operations using NumPy
- Efficient probability computation
- Caching of frequently used values

## Dependencies

- Python 3.7+
- pandas (for TSV file reading)
- numpy (for numerical computations)
- Standard library: json, collections, math, random, heapq, pathlib

## Expected Results

The implementation should generate:
- 100 sentences per model per approach (400 total per language)
- Diverse sentence structures
- Grammatically plausible sequences
- Comparative analysis showing differences between approaches

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure Lab4 directory is accessible
2. **Memory issues**: Reduce vocabulary size or sentence count
3. **Generation quality**: Adjust beam size or sentence length limits

### Performance Tips
- Use smaller datasets for faster experimentation
- Adjust beam size for quality vs speed tradeoff
- Monitor memory usage with large vocabularies

## Future Enhancements

- Neural language model integration
- Cross-language comparison
- Interactive sentence generation interface
- Advanced evaluation metrics (BLEU, perplexity)
- GPU acceleration for large models
