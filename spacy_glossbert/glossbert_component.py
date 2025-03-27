"""GlossBERT Word Sense Disambiguation spaCy component."""

import logging
from functools import lru_cache
from typing import Any, Callable, List, Optional, cast

import nltk
import torch
from nltk.corpus import wordnet as wn
from spacy.language import Language
from spacy.tokens import Doc, Token
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

# Register custom extension for tokens to store wordnet synsets
if not Token.has_extension("glossbert_synset"):
    Token.set_extension("glossbert_synset", default=None)


def has_glossbert_wsd(doc: Doc) -> bool:
    """Check if the document has been processed by the GlossBERT WSD component.

    Args:
        doc: The spaCy document to check

    Returns:
        True if the document has been processed by GlossBERT, False otherwise
    """
    return any(
        t.has_extension("glossbert_synset") and t._.glossbert_synset is not None
        for t in doc
    )


@lru_cache(maxsize=2048)
def get_synsets(word: str, pos: str) -> List[Any]:
    """Get WordNet synsets for a word with a given part of speech.

    Args:
        word: The word to get synsets for
        pos: The part of speech to filter synsets by

    Returns:
        A list of WordNet synsets
    """
    return cast(List[Any], wn.synsets(word, pos=pos))


class GlossBertWSD:
    """GlossBERT Word Sense Disambiguation component for spaCy."""

    def __init__(
        self,
        nlp: Language,
        name: str = "glossbert_wsd",
        pos_filter: Optional[List[str]] = None,
        supervision: bool = False,
        debug: bool = False,
        model_name: str = "kanishka/GlossBERT",
    ):
        """Initialize the GlossBERT WSD component.

        Args:
            nlp: The spaCy language pipeline
            name: The name of the component in the pipeline
            pos_filter: List of POS tags to process (defaults to ["NOUN", "VERB"])
            supervision: Whether to use supervision text in context
                (highlighting the target word)
            debug: Whether to print debug information
            model_name: HuggingFace model name/path for GlossBERT
        """
        self.name = name
        self.pos_filter = pos_filter or ["NOUN", "VERB"]
        self.supervision = supervision
        self.debug = debug
        self.model_name = model_name

        # Map spaCy POS tags to WordNet POS tags
        self.pos_map = {
            "NOUN": wn.NOUN,
            "VERB": wn.VERB,
            "ADJ": wn.ADJ,
            "ADV": wn.ADV,
            "PRON": wn.NOUN,  # Handle pronouns as nouns
        }

        # Load GlossBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Ensure NLTK wordnet is downloaded
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

    def __call__(self, doc: Doc) -> Doc:
        """Process a document with GlossBERT word sense disambiguation.

        Args:
            doc: The spaCy document to process

        Returns:
            The processed document with synsets attached to tokens
        """
        if self.debug:
            logger.debug("Processing document with GlossBERT WSD")

        for token in doc:
            if self.debug:
                logger.debug(f"Token.text: {token.text}")

            # Skip tokens with POS tags not in our filter
            if token.pos_ not in self.pos_filter:
                if self.debug:
                    logger.debug(
                        f"PoS tag {token.pos_} not in {self.pos_filter}. Skipping..."
                    )
                continue

            # Get corresponding WordNet POS tag
            wn_pos = self.pos_map.get(token.pos_)
            if not wn_pos:
                if self.debug:
                    logger.debug(f"No WordNet Pos found for {token.text}. Skipping...")
                continue

            if self.debug:
                logger.debug(f"WordNet Pos {wn_pos}")

            # Get WordNet synsets for this token
            synsets = get_synsets(token.text.lower(), pos=wn_pos)
            if not synsets:
                token._.glossbert_synset = None
                if self.debug:
                    logger.debug(f"No synsets found for {token.text}. Skipping...")
                continue

            if self.debug:
                logger.debug(f"{len(synsets)} Synsets: {synsets}")

            # Filter synsets to only those with matching POS
            valid_synsets = [synset for synset in synsets if synset.pos() == wn_pos]

            if not valid_synsets:
                if self.debug:
                    logger.debug(
                        f"No valid synsets found for {token.text}. Skipping..."
                    )
                continue

            if self.debug:
                logger.debug(f"{len(valid_synsets)} Valid synsets: {valid_synsets}")

            # Prepare inputs for each candidate sense
            inputs = []
            for synset in valid_synsets:
                gloss = synset.definition()

                # Create the context text, optionally highlighting the target word
                if self.supervision:
                    supervision_text = f'"{token.text.upper()}"' + token.whitespace_
                    text_before = doc.text[: token.idx]
                    text_after = doc.text[
                        token.idx + len(token.text + token.whitespace_) :
                    ]
                    context_text = text_before + supervision_text + text_after
                else:
                    context_text = doc.text

                # Format input for GlossBERT
                input_text = f"{context_text} [SEP] {gloss}"

                if self.debug:
                    logger.debug(f"synset: {synset}, Inputs: {input_text}")

                # Tokenize input for the model
                tokenized_input = self.tokenizer(
                    input_text, return_tensors="pt", truncation=True
                )
                inputs.append((synset, tokenized_input))

            # Score each candidate sense
            scores = []
            for sense, input_data in inputs:
                with torch.no_grad():
                    logits = self.model(**input_data).logits
                    score = torch.softmax(logits, dim=1)[0][1].item()
                    scores.append((sense, score))

            # Select the best-scoring sense
            best_sense = max(scores, key=lambda x: x[1])[0]
            if self.debug:
                logger.debug(f"Best Word Sense: {best_sense}")

            # Store the best sense in the token's custom attribute
            token._.glossbert_synset = best_sense

        return doc


@Language.factory(
    "glossbert_wsd",
    default_config={
        "pos_filter": ["NOUN", "VERB"],
        "supervision": False,
        "debug": False,
        "model_name": "kanishka/GlossBERT",
    },
)
def create_glossbert_wsd_component(
    nlp: Language,
    name: str,
    pos_filter: List[str],
    supervision: bool = False,
    debug: bool = False,
    model_name: str = "kanishka/GlossBERT",
) -> Callable[[Doc], Doc]:
    """Create a GlossBERT Word Sense Disambiguation component.

    Args:
        nlp: The spaCy language pipeline
        name: The name of the component in the pipeline
        pos_filter: List of POS tags to process (defaults to ["NOUN", "VERB"])
        supervision: Whether to use supervision text in context
        debug: Whether to print debug information
        model_name: HuggingFace model name/path for GlossBERT

    Returns:
        A callable component that can be added to the spaCy pipeline
    """
    return GlossBertWSD(
        nlp=nlp,
        name=name,
        pos_filter=pos_filter,
        supervision=supervision,
        debug=debug,
        model_name=model_name,
    )
