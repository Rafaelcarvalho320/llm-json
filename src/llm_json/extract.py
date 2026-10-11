"""The pipeline: candidate, repair, re-parse, report.

The design decision worth stating plainly is that nothing here is silent. A
parse that needed no help and a parse that was rescued by closing a cut-off
response both return a value, and most libraries in this space stop there. The
difference matters enormously to the caller - one is the model's answer, the
other is the first two thirds of it - so every :class:`Extraction` carries the
list of repairs that were necessary, and :func:`loads` can be told to refuse
the second case outright.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from llm_json.errors import JSONNotFound, TruncatedJSON, UnexpectedType
from llm_json.repairs import (
    COMMENTS,
    FENCED,
    PROSE,
    PY_LITERALS,
    QUOTES,
    TRAILING_COMMA,
    TRUNCATION,
    close_truncation,
    drop_trailing_commas,
    fenced_blocks,
    normalise_quotes,
    python_literals,
    strip_comments,
)
from llm_json.scanner import candidate_spans

#: ``(name, function)`` in application order. Cheapest and least destructive
#: first, so a response that is merely fenced never reaches quote rewriting.
PIPELINE: tuple[tuple[str, Callable[[str], str]], ...] = (
    (QUOTES, normalise_quotes),
    (COMMENTS, strip_comments),
    (TRAILING_COMMA, drop_trailing_commas),
    (PY_LITERALS, python_literals),
    (TRUNCATION, close_truncation),
)


@dataclass(frozen=True)
class Extraction:
    """A recovered JSON value and the story of how it was recovered."""

    #: The parsed value.
    value: Any
    #: The JSON text that was actually handed to :func:`json.loads`, after any
    #: repairs. Useful in a log: it is the difference between "the model is
    #: wrong" and "our parser is wrong", and guessing which from the original
    #: response is work nobody should have to do twice.
    text: str
    #: Where in the original text the value was found, as ``(start, end)``.
    #: ``None`` when it came out of a fenced block, whose offsets refer to the
    #: block rather than the response.
    span: tuple[int, int] | None
    #: Repair names, in the order they were applied.
    repairs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """Whether the text parsed with no repair at all."""
        return not self.repairs

    @property
    def truncated(self) -> bool:
        """Whether the value is the salvaged part of a cut-off response.

        The one flag worth checking before acting on the value: everything
        else this package repairs is cosmetic, and this one means fields are
        missing.
        """
        return TRUNCATION in self.repairs


def _parse(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except ValueError:
        return False, None


def _attempt(
    candidate: str,
    span: tuple[int, int] | None,
    origin: tuple[str, ...] = (),
) -> Extraction | None:
    """Run one candidate through the pipeline. ``None`` if nothing parsed.

    ``origin`` holds the repairs that getting hold of this candidate already
    required - unwrapping a code fence, discarding surrounding prose - so they
    appear in ``repairs`` alongside the textual ones.
    """
    stripped = candidate.strip()
    if not stripped:
        return None

    ok, value = _parse(stripped)
    if ok:
        return Extraction(value=value, text=stripped, span=span, repairs=origin)

    applied: list[str] = list(origin)
    current = stripped
    for name, repair in PIPELINE:
        repaired = repair(current)
        if repaired == current:
            continue
        current = repaired
        applied.append(name)
        ok, value = _parse(current)
        if ok:
            return Extraction(
                value=value, text=current, span=span, repairs=tuple(applied)
            )
    return None


Candidate = tuple[str, "tuple[int, int] | None", "tuple[str, ...]"]


def _candidates(text: str) -> Iterator[Candidate]:
    """Every place the JSON might be, best guess first.

    Fenced blocks come first because a model that was asked for JSON and wrote
    prose around it almost always fences the payload, and the prose itself can
    contain braces. The whole text is offered last, which covers the case of a
    response that is nothing but JSON.
    """
    seen: set[str] = set()
    whole = text.strip()

    for body in fenced_blocks(text):
        for start, end in candidate_spans(body):
            chunk = body[start:end]
            if chunk not in seen:
                seen.add(chunk)
                yield chunk, None, (FENCED,)

    for start, end in candidate_spans(text):
        chunk = text[start:end]
        if chunk in seen:
            continue
        seen.add(chunk)
        origin = () if chunk.strip() == whole else (PROSE,)
        yield chunk, (start, end), origin

    if whole and whole not in seen:
        yield whole, (0, len(text)), ()


def extract_all(text: str) -> Iterator[Extraction]:
    """Yield every JSON value recoverable from ``text``, in document order.

    For the common case of a model that was asked for one object and produced
    several - one per paragraph of its explanation - which is otherwise
    invisible, because taking the first and moving on looks like it worked.
    """
    for candidate, span, origin in _candidates(text):
        extraction = _attempt(candidate, span, origin)
        if extraction is not None:
            yield extraction


def extract(text: str) -> Extraction:
    """The first JSON value in ``text``, with the repairs it needed.

    Raises :class:`~llm_json.errors.JSONNotFound` when there is none, rather
    than returning ``None``: a caller who forgets to check ``None`` gets a
    confusing ``TypeError`` three frames later, and this is a mistake worth
    failing loudly on.
    """
    for extraction in extract_all(text):
        return extraction
    raise JSONNotFound(text)


def loads(
    text: str,
    *,
    expect: type | tuple[type, ...] | None = None,
    allow_truncated: bool = True,
) -> Any:
    """Like :func:`json.loads`, but for text written by a language model.

    :param expect: require this top-level type, e.g. ``dict``. Raises
        :class:`~llm_json.errors.UnexpectedType` otherwise. ``bool`` is not
        accepted as a shorthand for anything: it is a subclass of ``int``, and
        ``expect=int`` matching ``true`` would be a trap.
    :param allow_truncated: when ``False``, a value that only parsed because a
        cut-off response was closed raises
        :class:`~llm_json.errors.TruncatedJSON` instead of being returned.
        Worth setting on anything that writes to a database.
    """
    extraction = extract(text)
    if not allow_truncated and extraction.truncated:
        raise TruncatedJSON(extraction.value, extraction.text)
    if expect is not None and not _is_instance(extraction.value, expect):
        raise UnexpectedType(extraction.value, expect)
    return extraction.value


def _is_instance(value: Any, expected: type | tuple[type, ...]) -> bool:
    """``isinstance`` without ``bool`` sneaking through as ``int``."""
    wanted = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, bool) and bool not in wanted:
        return False
    return isinstance(value, wanted)


__all__ = [
    "PIPELINE",
    "Candidate",
    "Extraction",
    "extract",
    "extract_all",
    "loads",
]
