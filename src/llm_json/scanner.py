"""One pass over the text that every repair is built on.

Each repair in this package has an obvious regex version, and each obvious
regex version is wrong on real model output:

* Stripping ``//`` comments with a regex deletes the second half of every URL
  in the payload, because ``"https://example.com"`` contains ``//``.
* Dropping trailing commas with ``,\\s*}`` corrupts any string that happens to
  contain ``, }``, which is not rare in text written by a language model.
* Replacing ``True`` with ``true`` hits the word inside ``"note": "True story"``.

All three mistakes are the same mistake: editing JSON as if it were a flat
string. :func:`classify` resolves it once, by labelling every character as
code, string or comment in a single left-to-right pass. A ``//`` seen while
inside a string is not a comment; a quote seen while inside a comment does not
open a string. Everything downstream asks the labels instead of guessing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

#: Character labels produced by :func:`classify`.
CODE = 0
STRING = 1
COMMENT = 2

OPENERS = {"{": "}", "[": "]"}
CLOSERS = {"}": "{", "]": "["}
WHITESPACE = " \t\r\n"
KEYWORDS = ("true", "false", "null")


def classify(text: str) -> list[int]:
    """Label every character as :data:`CODE`, :data:`STRING` or :data:`COMMENT`.

    The quotes delimiting a string are labelled ``STRING`` along with its
    contents, so a caller can rebuild the text by filtering on the labels
    without having to special-case the delimiters.

    An unterminated string runs to the end of the input rather than being
    rejected. That is the normal shape of a response cut off at the token
    limit, and :func:`walk` needs to see it to close it.
    """
    kinds = [CODE] * len(text)
    index = 0
    size = len(text)

    while index < size:
        char = text[index]

        if char == '"':
            kinds[index] = STRING
            index += 1
            while index < size:
                kinds[index] = STRING
                if text[index] == "\\":
                    # The escaped character belongs to the string whatever it
                    # is, including a quote, which is the whole point.
                    if index + 1 < size:
                        kinds[index + 1] = STRING
                    index += 2
                    continue
                if text[index] == '"':
                    index += 1
                    break
                index += 1
            continue

        if char == "/" and index + 1 < size and text[index + 1] == "/":
            while index < size and text[index] != "\n":
                kinds[index] = COMMENT
                index += 1
            continue

        if char == "/" and index + 1 < size and text[index + 1] == "*":
            kinds[index] = kinds[index + 1] = COMMENT
            index += 2
            while index < size:
                kinds[index] = COMMENT
                if text[index] == "*" and index + 1 < size and text[index + 1] == "/":
                    kinds[index + 1] = COMMENT
                    index += 2
                    break
                index += 1
            continue

        index += 1

    return kinds


def candidate_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` for each top-level ``{``/``[`` in the text.

    Prose around the JSON is handled here rather than by a "find the first
    brace" rule, because the first brace is frequently not the answer::

        Use {placeholder} syntax. Here is the result: {"ok": true}

    So every structural opener that is not already inside a yielded span is
    offered as a candidate, and the caller tries them in order until one
    parses. The last candidate of a truncated response ends at the end of the
    input, which is exactly what :func:`walk` is given to close.
    """
    kinds = classify(text)
    index = 0
    size = len(text)

    while index < size:
        if kinds[index] == CODE and text[index] in OPENERS:
            end = _matching_end(text, kinds, index)
            yield index, end
            index = end
            continue
        index += 1


def _matching_end(text: str, kinds: list[int], start: int) -> int:
    """Index just past the closer that balances ``text[start]``.

    Returns ``len(text)`` when it never closes, which is the truncated case.
    """
    depth = 0
    for index in range(start, len(text)):
        if kinds[index] != CODE:
            continue
        char = text[index]
        if char in OPENERS:
            depth += 1
        elif char in CLOSERS:
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


class _Frame:
    """One open container, and how much of it is known to be complete."""

    __slots__ = ("boundary", "expecting_value", "kind", "partial_child", "values")

    def __init__(self, kind: str, start: int) -> None:
        self.kind = kind
        #: Index just past this container's last complete member or element.
        self.boundary = start
        #: How many members or elements completed inside it.
        self.values = 0
        #: Whether a child container is open inside it right now, which makes
        #: this container's current member or element unfinished.
        self.partial_child = False
        #: Objects alternate key and value. Arrays are values throughout.
        self.expecting_value = kind == "["


class Walk:
    """Where a truncated document can be cut, and what to close it with."""

    __slots__ = ("complete", "safe_end", "safe_stack", "values")

    def __init__(
        self, complete: bool, safe_end: int, safe_stack: str, values: int
    ) -> None:
        self.complete = complete
        #: Index just past the chosen cut point.
        self.safe_end = safe_end
        #: Containers still open at the cut, outermost first.
        self.safe_stack = safe_stack
        #: Complete members or elements inside the container being cut.
        self.values = values

    @property
    def salvageable(self) -> bool:
        """Whether closing here recovers anything the model actually sent.

        Without this test, closing is not a repair but a fabrication: the
        prose ``Use {placeholder} syntax`` has a balanced brace, fails to
        parse, and then "closes" into a perfectly valid ``{}`` that outranks
        the real object later in the same reply.
        """
        return self.values > 0

    def closed(self, text: str) -> str:
        """``text`` cut at :attr:`safe_end`, with the missing closers appended."""
        closers = "".join(OPENERS[char] for char in reversed(self.safe_stack))
        return text[: self.safe_end] + closers


def walk(text: str) -> Walk:
    """Find the best cut point in a possibly truncated JSON document.

    The contract, which the rest of the rules follow from:

        **An object keeps the members that arrived. An array keeps only
        complete elements.**

    That asymmetry is not arbitrary. A truncated reply's outer object is an
    envelope, and the three fields that made it through are worth having. An
    unfinished element *inside* a list is a record with fields missing, and
    handing it over mixed in with the good ones is how a half-read row reaches
    a database looking exactly like a whole one::

        {"a": 1, "b":                      ->  {"a": 1}
        [{"id": 1}, {"id": 2}, {"id": 3,   ->  [{"id": 1}, {"id": 2}]

    Two smaller rules, both learned by getting them wrong first:

    **A string is only safe to cut after when it was a value.** In
    ``{"a": 1, "b":`` the last complete token is the key ``"b"``; closing
    there gives ``{"a": 1, "b"}``, which is not JSON.

    **A number touching the end of the input is not complete.** ``12`` parses,
    but it is also a prefix of ``125``, and keeping it invents a value the
    model never finished writing. Keywords do not have that problem: nothing
    valid starts with ``true`` and continues, so ``true`` at the end is kept.
    """
    kinds = classify(text)
    size = len(text)
    frames: list[_Frame] = []

    def complete_value(end: int) -> None:
        """Record that the current container just finished a member or element."""
        if not frames:
            return
        frame = frames[-1]
        frame.boundary = end
        frame.values += 1
        frame.partial_child = False

    index = 0
    while index < size:
        if kinds[index] == COMMENT:
            index += 1
            continue

        char = text[index]

        if kinds[index] == STRING:
            end, terminated = _string_end(text, index)
            if not terminated:
                # Ran off the end mid-string: nothing past the last recorded
                # boundary can be trusted.
                break
            in_key_position = (
                bool(frames)
                and frames[-1].kind == "{"
                and not frames[-1].expecting_value
            )
            if not in_key_position:
                complete_value(end)
            index = end
            continue

        if char in OPENERS:
            if frames:
                frames[-1].partial_child = True
            frames.append(_Frame(char, index + 1))
            index += 1
            continue

        if char in CLOSERS:
            if not frames or frames[-1].kind != CLOSERS[char]:
                break
            frames.pop()
            complete_value(index + 1)
            index += 1
            continue

        if char == ":":
            if frames:
                frames[-1].expecting_value = True
            index += 1
            continue

        if char == ",":
            if frames and frames[-1].kind == "{":
                frames[-1].expecting_value = False
            index += 1
            continue

        if char in WHITESPACE:
            index += 1
            continue

        scalar_end = _scalar_end(text, index)
        if scalar_end is None:
            break
        token = text[index:scalar_end]
        if scalar_end == size and _is_number(token):
            # Could be the first digits of a longer number. Drop it.
            break
        if _is_number(token) or token in KEYWORDS:
            complete_value(scalar_end)
            index = scalar_end
            continue
        break

    if index >= size and not frames:
        return Walk(True, size, "", 1)
    return _cut(frames)


def _cut(frames: list[_Frame]) -> Walk:
    """Choose which container to cut, applying the object/array contract."""
    if not frames:
        # Nothing was ever opened, so there is nothing to close and nothing to
        # salvage: the text was prose, or a bare scalar that did not parse.
        return Walk(False, 0, "", 0)

    chosen = len(frames) - 1
    for position, frame in enumerate(frames):
        # The outermost array with an unfinished element decides the cut: that
        # element, and everything inside it, is dropped.
        if frame.kind == "[" and frame.partial_child:
            chosen = position
            break

    # Nothing completed at that depth, so back out to an ancestor that did.
    # Otherwise `{"a": 1, "b": {"c":` salvages nothing, when `{"a": 1}` was
    # right there.
    while chosen > 0 and frames[chosen].values == 0:
        chosen -= 1

    frame = frames[chosen]
    stack = "".join(item.kind for item in frames[: chosen + 1])
    return Walk(False, frame.boundary, stack, frame.values)


def _string_end(text: str, start: int) -> tuple[int, bool]:
    """``(index just past the string, whether it was closed)``.

    Escape-aware on purpose: a string ending in ``\\"`` has a quote as its last
    character and is still unterminated, and a "does it end with a quote" test
    reads it as closed and then loses everything after it.
    """
    index = start + 1
    size = len(text)
    while index < size:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return index + 1, True
        index += 1
    return size, False


def _scalar_end(text: str, start: int) -> int | None:
    index = start
    size = len(text)
    while index < size and text[index] not in ",}]" and text[index] not in WHITESPACE:
        index += 1
    return index if index > start else None


#: What JSON calls a number. ``float()`` is not a substitute: it accepts
#: ``nan``, ``infinity`` and ``1_0``, none of which are valid JSON.
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?\Z")


def _is_number(token: str) -> bool:
    return _NUMBER.match(token) is not None
