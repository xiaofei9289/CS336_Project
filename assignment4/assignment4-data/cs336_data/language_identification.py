"""Language identification with the fastText lid.176 model."""

from functools import cache

import fasttext

from cs336_data.common import get_shared_assets_path

_LABEL_PREFIX = "__label__"


@cache
def _model() -> fasttext.FastText._FastText:
    path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    return fasttext.load_model(str(path))


def identify_language(text: str) -> tuple[str, float]:
    """Return the top language code and a confidence score in [0, 1]."""
    # fastText treats newlines as document boundaries.
    cleaned = " ".join(text.split())
    labels, scores = _model().predict(cleaned, k=1)
    language = labels[0].removeprefix(_LABEL_PREFIX)
    if language in {"zh-cn", "zh-tw", "zh-hans", "zh-hant"}:
        language = "zh"
    return language, float(scores[0])


def is_english(text: str) -> bool:
    """Return whether fastText calls the text English with probability >= 0.7."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return False
    language, probability = identify_language(cleaned)
    return language == "en" and probability >= 0.7
