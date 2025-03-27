#!/usr/bin/env python3
"""Example usage of the GlossBERT WSD spaCy component."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import spacy

from spacy_glossbert import get_synset_info


def main() -> None:
    """Demonstrate basic usage of the GlossBERT WSD component."""
    # Load spaCy with the English model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Add GlossBERT WSD component to the pipeline
    print("Adding GlossBERT WSD component...")
    nlp.add_pipe(
        "glossbert_wsd",
        config={"pos_filter": ["NOUN", "VERB"], "supervision": False, "debug": True},
        last=True,
    )

    # Process a text with ambiguous words
    text = "He went to the bank to deposit money after falling in love."
    print(f'\nProcessing: "{text}"')
    doc = nlp(text)

    # Print the results
    print("\nDisambiguated word senses:")
    print("--------------------------")
    for token in doc:
        synset = token._.glossbert_synset
        if synset:
            print(
                f"{token.text}: {token.pos_} -- {synset.name()} - {synset.definition()}"
            )

    # Get structured information
    print("\nStructured sense information:")
    print("----------------------------")
    sense_info = get_synset_info(doc)
    for info in sense_info:
        print(
            f"{info['text']} ({info['pos']}): {info['synset']} - {info['definition']}"
        )

    # Note about visualization
    print("\nTo visualize the results, use:")
    print("  from spacy_glossbert import visualize_wsd")
    print("  visualize_wsd(doc)")


if __name__ == "__main__":
    main()
