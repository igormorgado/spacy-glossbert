"""Utility functions for GlossBERT WSD component."""

from typing import Any, Dict, List

from spacy import displacy
from spacy.tokens import Doc


def prepare_entities_for_visualization(doc: Doc) -> List[Dict[str, Any]]:
    """Prepare entities for visualization with displaCy.

    Args:
        doc: The spaCy document processed with GlossBERT WSD

    Returns:
        A list of entity dictionaries for displaCy visualization
    """
    entities = []
    for token in doc:
        synset = token._.get("glossbert_synset")
        if synset:
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


def get_synset_info(doc: Doc) -> List[Dict[str, str]]:
    """Get information about disambiguated word senses in a document.

    Args:
        doc: The spaCy document processed with GlossBERT WSD

    Returns:
        A list of dictionaries with token text, POS, synset name, and definition
    """
    results = []

    for token in doc:
        synset = token._.get("glossbert_synset")
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
