"""Heuristic quality filters from the Gopher paper."""


def _has_letter(word: str) -> bool:
    return any(character.isalpha() for character in word)


def gopher_quality_filter(text: str) -> bool:
    """Return whether ``text`` passes the Gopher quality rules.

    A document is rejected when it has fewer than 50 or more than 100,000
    words, a mean word length outside 3 to 10 characters, more than 30% of
    lines ending in ``...``, or fewer than 80% of words containing a letter.
    """
    words = text.split()
    if len(words) < 50 or len(words) > 100_000:
        return False

    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines() or [text]
    ellipsis_lines = sum(line.endswith("...") for line in lines)
    if ellipsis_lines / len(lines) > 0.3:
        return False

    alphabetic_words = sum(_has_letter(word) for word in words)
    if alphabetic_words / len(words) < 0.8:
        return False
    return True
