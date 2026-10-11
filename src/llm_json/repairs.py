"""The individual repairs, in the order :mod:`llm_json.extract` applies them.

Every function takes the candidate text and returns it changed, or returns it
unchanged when it had nothing to do. The caller re-parses after each one and
stops at the first success, so a response that only needed its fences stripped
is never also put through quote normalisation. That ordering is what keeps the
list of repairs on an :class:`~llm_json.extract.Extraction` honest: it reports
what was actually necessary, not what was attempted.
"""

from __future__ import annotations

import re

from llm_json.scanner import CLOSERS, CODE, COMMENT, WHITESPACE, classify, walk

#: Names recorded on an :class:`~llm_json.extract.Extraction`. They are part of
#: the public surface: callers switch on them, so they are defined once here.
FENCED = "unfenced"
PROSE = "removed-prose"
QUOTES = "normalised-quotes"
COMMENTS = "removed-comments"
TRAILING_COMMA = "removed-trailing-comma"
PY_LITERALS = "python-literals"
TRUNCATION = "closed-truncation"

#: Curly quotes, as emitted by a model that has been asked for JSON inside
#: prose and has helpfully typeset it.
CURLY_DOUBLE = "“”„‟"
CURLY_SINGLE = "‘’‚‛"

_FENCE = re.compile(
    r"```[ \t]*(?:json|json5|javascript|js)?[ \t]*\r?\n(?P<body>.*?)(?:```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_PY_LITERAL = re.compile(r"\b(True|False|None)\b")
_PY_TO_JSON = {"True": "true", "False": "false", "None": "null"}


def fenced_blocks(text: str) -> list[str]:
    """Bodies of the ```` ``` ```` blocks in ``text``, outermost first.

    A block with no closing fence still yields its body. That is the ordinary
    shape of a truncated response: the model opened the fence, started the
    object and ran out of tokens, so insisting on a matching close throws away
    the only content there is.
    """
    return [match.group("body") for match in _FENCE.finditer(text)]


def normalise_quotes(text: str) -> str:
    """Turn typographic and single quotes into the only quote JSON accepts.

    Guarded, and the guard is the interesting part: this runs **only** when the
    text contains no straight double quote at all. If the model delimited its
    strings properly, then a curly quote inside one is content - someone's
    words - and rewriting it corrupts the value while fixing nothing.
    """
    if '"' in text:
        return text

    out = text
    for char in CURLY_DOUBLE:
        out = out.replace(char, '"')
    if '"' in out:
        return out

    # No double quotes of any shape: the model used single ones.
    for char in CURLY_SINGLE:
        out = out.replace(char, "'")
    return out.replace("'", '"')


def strip_comments(text: str) -> str:
    """Drop ``//`` and ``/* */`` comments, and nothing that merely looks like one.

    The reason this is not a regex: ``{"url": "https://example.com"}`` contains
    ``//`` inside a string, and a regex-based stripper turns it into
    ``{"url": "https:}`` - valid-looking, silently wrong, and the kind of bug
    that is found in production rather than in review.
    """
    kinds = classify(text)
    if COMMENT not in kinds:
        return text
    return "".join(
        char for char, kind in zip(text, kinds, strict=True) if kind != COMMENT
    )


def drop_trailing_commas(text: str) -> str:
    """Remove a comma that is followed only by whitespace and a closer."""
    kinds = classify(text)
    doomed: set[int] = set()

    for index, char in enumerate(text):
        if char != "," or kinds[index] != CODE:
            continue
        probe = index + 1
        while probe < len(text) and text[probe] in WHITESPACE:
            probe += 1
        if probe < len(text) and kinds[probe] == CODE and text[probe] in CLOSERS:
            doomed.add(index)

    if not doomed:
        return text
    return "".join(char for index, char in enumerate(text) if index not in doomed)


def python_literals(text: str) -> str:
    """``True``/``False``/``None`` to their JSON spellings, outside strings.

    A model asked for JSON by a Python program will sometimes answer in Python,
    and the word ``True`` also appears in English. Only code positions are
    rewritten, so ``{"note": "True story"}`` is left exactly as it is.
    """
    kinds = classify(text)
    pieces: list[str] = []
    cursor = 0

    for match in _PY_LITERAL.finditer(text):
        if kinds[match.start()] != CODE:
            continue
        pieces.append(text[cursor : match.start()])
        pieces.append(_PY_TO_JSON[match.group(1)])
        cursor = match.end()

    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def close_truncation(text: str) -> str:
    """Close the containers a cut-off response left open.

    Trims back to the last position where closing was valid, then appends the
    closers. See :func:`llm_json.scanner.walk` for which positions count: the
    short version is that a half-written key, a bare container opener, and a
    number sitting against the end of the input are all dropped rather than
    guessed at.

    Refuses outright when nothing complete was seen before the cut. Returning
    ``{}`` for a response that died inside its first key is not a recovered
    value but an invented one, and downstream it reads as "the model found
    nothing" rather than "the model was interrupted".
    """
    scan = walk(text)
    if scan.complete or not scan.salvageable:
        return text
    return scan.closed(text)
