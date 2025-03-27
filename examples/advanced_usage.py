#!/usr/bin/env python3
"""Advanced usage of the GlossBERT WSD spaCy component."""

import logging
from typing import Dict, List

import spacy
from spacy import displacy
from spacy.tokens import Doc

from spacy_glossbert import get_synset_info


def setup_custom_pipeline() -> spacy.Language:
    """Set up a custom spaCy pipeline with GlossBERT WSD.

    Returns:
        A configured spaCy Language object
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load spaCy with the English model
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Configure and add GlossBERT WSD component with custom settings
    logger.info("Adding GlossBERT WSD component with custom settings...")
    config = {
        "pos_filter": ["NOUN", "VERB", "ADJ"],  # Add adjectives
        "supervision": True,
        "model_name": "kanishka/GlossBERT",
    }
    nlp.add_pipe("glossbert_wsd", config=config, last=True)

    # Log pipeline components
    logger.info(f"Pipeline components: {nlp.pipe_names}")

    return nlp


def custom_visualization(doc: Doc, colors: Dict[str, str] = None) -> None:
    """Custom visualization for WSD results.

    Args:
        doc: The spaCy document processed with GlossBERT WSD
        colors: Optional dictionary mapping POS tags to colors
    """
    # Default colors if none provided
    if colors is None:
        colors = {
            "NOUN": "#ff6666",  # Red
            "VERB": "#66ff66",  # Green
            "ADJ": "#6666ff",  # Blue
        }

    # Prepare entities for visualization
    entities = []
    for token in doc:
        synset = token._.glossbert_synset
        if synset:
            # Map synset POS to color
            pos_map = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
            pos = pos_map.get(synset.pos(), "OTHER")
            color = colors.get(pos, "#999999")  # Default gray

            entities.append(
                {
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "label": synset.name(),
                    "kb_id": synset.definition(),
                    "color": color,
                }
            )

    # Add entities to doc for visualization
    doc.user_data["ents"] = entities

    # Custom options for displaCy
    options = {
        "ents": [entity["label"] for entity in entities],
        "colors": {entity["label"]: entity["color"] for entity in entities},
    }

    # Render with displaCy
    displacy.render(doc, style="ent", options=options)


def analyze_multiple_senses(text: str, nlp: spacy.Language) -> List[Dict]:
    """Analyze multiple potential senses for each word.

    Args:
        text: The text to analyze
        nlp: The spaCy Language object with GlossBERT WSD

    Returns:
        A list of dictionaries with word and sense information
    """
    # Get the GlossBERT component
    component = nlp.get_pipe("glossbert_wsd")

    # Process the text
    doc = nlp(text)

    # Collect multiple senses for each word
    results = []
    for token in doc:
        if token.pos_ not in component.pos_filter:
            continue

        # Get WordNet POS
        wn_pos = component.pos_map.get(token.pos_)
        if not wn_pos:
            continue

        # Get all synsets
        synsets = component.get_synsets(token.text.lower(), pos=wn_pos)

        # Only consider synsets with matching POS
        valid_synsets = [synset for synset in synsets if synset.pos() == wn_pos]

        if valid_synsets:
            # Get the disambiguated sense
            disambiguated = token._.glossbert_synset

            # Add to results
            results.append(
                {
                    "word": token.text,
                    "pos": token.pos_,
                    "disambiguated": disambiguated.name() if disambiguated else None,
                    "potential_senses": [
                        {"name": s.name(), "definition": s.definition()}
                        for s in valid_synsets
                    ],
                }
            )

    return results


def main() -> None:
    """Run the advanced example."""
    # Set up pipeline
    nlp = setup_custom_pipeline()

    # Define a text with ambiguous words
    text = "The bass player adjusted the bass on his amplifier while fishing for bass."
    print(f'\nProcessing: "{text}"\n')

    # Process the text
    doc = nlp(text)

    # Print basic results
    print("Disambiguated Senses:")
    print("---------------------")
    for info in get_synset_info(doc):
        print(
            f"{info['text']} ({info['pos']}): {info['synset']} - {info['definition']}"
        )

    # Analyze multiple potential senses
    print("\nAll Potential Senses:")
    print("---------------------")
    sense_analysis = analyze_multiple_senses(text, nlp)
    for item in sense_analysis:
        print(f"\n{item['word']} ({item['pos']}):")
        print(f"  ✅ DISAMBIGUATED: {item['disambiguated']}")
        print("  Potential senses:")
        for i, sense in enumerate(item["potential_senses"], 1):
            prefix = "→" if sense["name"] == item["disambiguated"] else " "
            print(f"  {prefix} {i}. {sense['name']}: {sense['definition']}")

    # Note about visualization
    print("\nTo visualize with custom colors, use custom_visualization(doc)")


if __name__ == "__main__":
    main()
