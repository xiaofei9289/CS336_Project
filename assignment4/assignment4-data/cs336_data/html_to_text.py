"""HTML byte strings to plain text."""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Decode raw HTML bytes and extract visible text.

    UTF-8 is tried first. If those bytes are not valid UTF-8, the encoding
    is detected and the decode is retried.
    """
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes) or "utf-8"
        html = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html)
