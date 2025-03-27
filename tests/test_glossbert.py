"""Tests for the GlossBERT spaCy component."""

import os
import sys
import nltk

sys.path.insert(0, os.path.abspath("."))

import spacy

# Explicitly register the factory to ensure it's available
from spacy.language import Language
from spacy.tokens import Token

# Import the component and make sure it's registered
from spacy_glossbert import (
    create_glossbert_wsd_component,
)

# Ensure NLTK wordnet is downloaded
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

if not Language.has_factory("glossbert_wsd"):
    Language.factory("glossbert_wsd")(create_glossbert_wsd_component)


def test_glossbert_component_creation():
    """Test that the GlossBERT component can be created."""
    nlp = spacy.blank("en")

    # The component should be registered by importing spacy_glossbert
    # Just check that the component can be registered without errors
    glossbert = nlp.add_pipe(
        "glossbert_wsd",
        last=True,
        config={"model_name": "kanishka/GlossBERT", "pos_filter": ["NOUN", "VERB"]},
    )
    assert "glossbert_wsd" in nlp.pipe_names
    assert glossbert is not None


def test_has_glossbert_extension():
    """Test that tokens have the glossbert_synset extension."""
    # Setup the extension by creating a component instance
    nlp = spacy.blank("en")
    nlp.add_pipe("glossbert_wsd", last=True)

    # Now check that the extension is registered on tokens
    assert Token.has_extension("glossbert_synset")
