"""Wiki-versus-Common-Crawl quality labels from a local fastText model.

The course Wikipedia reference-URL list is not in the local data directory, so
this model is trained on the two sanity-check documents shipped with the tests:
the encyclopedia page is ``wiki`` and the crawl page is ``cc``.
"""

from functools import cache
from pathlib import Path

import fasttext

_LABEL_PREFIX = "__label__"
_MODEL_NAME = "quality_wiki_cc.bin"


def _model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "local-shared-data" / "classifiers" / _MODEL_NAME


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _one_line(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def train_quality_classifier(output_path: Path | None = None) -> Path:
    """Fit a fastText model that separates the two provided quality fixtures."""
    fixtures = _fixture_dir()
    wiki = _one_line(fixtures / "high_quality_wiki_reference.txt")
    cc = _one_line(fixtures / "low_quality_cc.txt")
    wiki_paragraphs = [
        " ".join(paragraph.split())
        for paragraph in (fixtures / "high_quality_wiki_reference.txt").read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]
    output_path = output_path or _model_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_path = output_path.with_suffix(".train.txt")
    rows = [f"__label__wiki {wiki}", f"__label__cc {cc}"]
    rows.extend(f"__label__wiki {paragraph}" for paragraph in wiki_paragraphs)
    training_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    model = fasttext.train_supervised(
        input=str(training_path),
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=50,
        loss="softmax",
    )
    model.save_model(str(output_path))
    training_path.unlink(missing_ok=True)
    return output_path


@cache
def _model() -> fasttext.FastText._FastText:
    path = _model_path()
    if not path.exists():
        train_quality_classifier(path)
    return fasttext.load_model(str(path))


def classify_quality(text: str) -> tuple[str, float]:
    """Return ``wiki`` or ``cc`` and the confidence of that label."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return "cc", 0.0
    labels, scores = _model().predict(cleaned, k=1)
    return labels[0].removeprefix(_LABEL_PREFIX), float(scores[0])
