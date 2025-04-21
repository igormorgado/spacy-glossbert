"""Utility functions for GlossBERT WSD component."""

from typing import Any, Dict, List

from spacy import displacy
from spacy.tokens import Doc, Token

import nltk
from nltk.corpus import wordnet as wn


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


def prepare_data_for_visualization(doc: Doc, style: str = "ent") -> List[Dict[str, Any]]:
    """Prepare data for visualization with displaCy.

    Args:
        doc: The spaCy document processed with GlossBERT WSD

    Returns:
        A list of entity dictionaries for displaCy visualization
    """
    entities = []
    for token in doc:
        synset = get_synset(token)
        if synset:
            if style == "span":
                entities.append(
                    {
                        "start_token": token.i,
                        "end_token": token.i + 1,
                        "label": synset.name(),
                    }
                )
            elif style == "ent":
                entities.append(
                    {
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "label": synset.name(),
                    }
                )


    return entities


def visualize_wsd(
    doc: Doc,
    style: str = "ent", 
    title: str | None = None, 
    options: dict | None = None
) -> None:
    """Visualize word sense disambiguation results.

    Args:
        doc: The spaCy document processed with GlossBERT WSD
        style: The visualization style to use
        title: A title for the generated plot
        options: options to pass to displacy

    Returns:
        None
    """
    if style not in ["ent", "span"]:
        raise ValueError("style must be `ent` or `span`")

    # Add entities to the document's user data
    if style == "span":
        data = {
            "text": doc.text,
            "spans": prepare_data_for_visualization(doc, style=style),
            "tokens": [token.text for token in doc],
        }
    elif style == "ent":
        data = {
            "text": doc.text,
            "ents": prepare_data_for_visualization(doc, style=style),
        }

    if title is not None:
        data["title"] = title

    # Visualize with displaCy
    displacy_params = {
        "style": style,
        "manual": True,
    }
    if options is not None:
        displacy_params["options"] = options

    displacy.render(data, **displacy_params)

