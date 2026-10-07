"""NSFW and toxic-speech labels from the Dolma Jigsaw fastText models."""

from functools import cache

import fasttext

from cs336_data.common import get_shared_assets_path

_LABEL_PREFIX = "__label__"
_NSFW_MODEL = "dolma_fasttext_nsfw_jigsaw_model.bin"
_TOXIC_MODEL = "dolma_fasttext_hatespeech_jigsaw_model.bin"


@cache
def _model(filename: str) -> fasttext.FastText._FastText:
    path = get_shared_assets_path() / "classifiers" / filename
    return fasttext.load_model(str(path))


def _classify(filename: str, text: str) -> tuple[str, float]:
    # fastText treats newlines as document boundaries.
    cleaned = " ".join(text.split())
    labels, scores = _model(filename).predict(cleaned, k=1)
    label = labels[0].removeprefix(_LABEL_PREFIX)
    return label, float(scores[0])


def classify_nsfw(text: str) -> tuple[str, float]:
    """Return ``nsfw`` or ``non-nsfw`` and the confidence of that label."""
    return _classify(_NSFW_MODEL, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Return ``toxic`` or ``non-toxic`` and the confidence of that label."""
    return _classify(_TOXIC_MODEL, text)
