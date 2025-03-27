# Spacy-GlossBert

A spaCy pipeline component for word sense disambiguation using the GlossBERT model.

## Overview

This package provides a spaCy component that performs Word Sense Disambiguation (WSD) using the GlossBERT model. 
GlossBERT leverages BERT's contextual embeddings to disambiguate word senses by comparing the context with WordNet sense definitions (glosses).

## Installation

```bash
pip install spacy-glossbert
```

You'll also need to download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
import spacy
from spacy_glossbert import has_glossbert_wsd, get_synset_info

# Load spaCy with the GlossBERT component
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("glossbert_wsd", last=True)

# Process a text
doc = nlp("He went to the bank to deposit money.")

# Check if the document has been processed with GlossBERT
if has_glossbert_wsd(doc):
    # Print disambiguated senses
    for token in doc:
        synset = token._.glossbert_synset
        if synset:
            print(f"{token.text}: {token.pos_} -- {synset.name()} - {synset.definition()}")

    # Alternative: Get all sense information as a list of dictionaries
    senses = get_synset_info(doc)
    for sense in senses:
        print(f"{sense['text']}: {sense['synset']} - {sense['definition']}")
```

### Visualization

The package includes visualization utilities using spaCy's displaCy:

```python
from spacy_glossbert import visualize_wsd

# Visualize the disambiguated senses
visualize_wsd(doc)
```

### Configuration Options

```python
# Configure the component
nlp.add_pipe(
    "glossbert_wsd",
    config={
        "pos_filter": ["NOUN", "VERB", "ADJ"],  # Part-of-speech tags to process
        "supervision": True,  # Highlight the target word in context
        "debug": False,  # Print debug information
        "model_name": "kanishka/GlossBERT"  # HuggingFace model name/path
    }
)
```

## How It Works

The GlossBERT component:

1. Identifies tokens with POS tags specified in `pos_filter`
2. Retrieves WordNet synsets (possible senses) for each token
3. For each candidate sense, creates an input of the form: `"context [SEP] definition"`
4. Scores each sense using the GlossBERT model
5. Assigns the highest-scoring sense to each token

## License

This project is licensed under the GNU General Public License v2.0 (GPLv2).

## Credits

- GlossBERT model: [Huang et al](https://aclanthology.org/D19-1355/)
- spaCy: [Explosion AI](https://spacy.io/)
- WordNet: [Princeton University](https://wordnet.princeton.edu/) 

## Author

Igor Morgado <morgado.igor@gmail.com> 