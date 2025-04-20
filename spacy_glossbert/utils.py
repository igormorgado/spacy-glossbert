"""Utility functions for GlossBERT WSD component."""

from typing import Any, Dict, List

from spacy import displacy
from spacy.tokens import Doc, Token

import nltk
from nltk.corpus import wordnet as wn


def prepare_entities_for_visualization(doc: Doc) -> List[Dict[str, Any]]:
    """Prepare entities for visualization with displaCy.

    Args:
        doc: The spaCy document processed with GlossBERT WSD

    Returns:
        A list of entity dictionaries for displaCy visualization
    """
    entities = []
    for token in doc:
        synset_name = token._.get("glossbert_synset")
        if synset_name:
            synset = wn.synset(synset_name)
            entities.append(
                {
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "label": synset.name(),
                }
            )

    return entities


def visualize_wsd(doc: Doc, style: str = "ent") -> None:
    """Visualize word sense disambiguation results.

    Args:
        doc: The spaCy document processed with GlossBERT WSD
        style: The visualization style to use

    Returns:
        None
    """
    # Prepare entities based on glossbert synsets
    entities = prepare_entities_for_visualization(doc)

    # Add entities to the document's user data
    doc.user_data["ents"] = entities

    # Visualize with displaCy
    displacy.render(doc, style=style)


def get_synset_from_name(synset_name: str | None) -> nltk.corpus.reader.wordnet.Synset | None:
    """Get synset object from a synset name.

    Args:
        synset_name: A synset name as string.

    Returns:
        synset: a wordnet synset object. (or None if no synset is associated).

    Raises:
        WordNetError: If given synset_name does not exist.
    """
    if synset_name is None:
        return None
    return wn.synset(synset_name)


def get_synset(token: Token) -> nltk.corpus.reader.wordnet.Synset | None:
    """Get spacy token synset object.

    Args:
        token: a spaCy token.

    Returns:
        synset: a wordnet synset object. (or None if no synset is associated)
    """
    synset_name = token._.get("glossbert_synset")
    if synset_name:
        return get_synset_from_name(synset_name)
    else:
        return None


def get_synset_info(doc: Doc) -> List[Dict[str, str]]:
    """Get information about disambiguated word senses in a document.

    Args:
        doc: The spaCy document processed with GlossBERT WSD

    Returns:
        A list of dictionaries with token text, POS, synset name, and definition
    """
    results = []

    for token in doc:
        synset = get_synset(token)
        if synset:
            results.append(
                {
                    "text": token.text,
                    "pos": token.pos_,
                    "synset": synset.name(),
                    "definition": synset.definition(),
                }
            )

    return results
