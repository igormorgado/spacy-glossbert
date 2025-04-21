"""GlossBERT Word Sense Disambiguation spaCy component."""

# Ensure the component is registered with spaCy
from spacy.language import Language

from .glossbert_component import (
    GlossBertWSD,
    create_glossbert_wsd_component,
    has_glossbert_wsd,
)
from .utils import (
    get_synset,
    get_synset_from_name,
    get_synset_info,
    prepare_entities_for_visualization,
    visualize_wsd,
)

if not Language.has_factory("glossbert_wsd"):
    Language.factory(
        "glossbert_wsd",
        default_config={
            "pos_filter": ["NOUN", "VERB"],
            "supervision": False,
            "model_name": "kanishka/GlossBERT",
        },
        requires=['token.tag'],
        assigns=['token._.glossbert_synset'],
    )(create_glossbert_wsd_component)
    # Language.factory("glossbert_wsd")(create_glossbert_wsd_component)

__all__ = [
    "GlossBertWSD",
    "create_glossbert_wsd_component",
    "has_glossbert_wsd",
    "get_synset",
    "get_synset_info",
    "prepare_entities_for_visualization",
    "visualize_wsd",
]
