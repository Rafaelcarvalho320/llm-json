"""llm-json: get a usable JSON value out of what a language model actually sent.

A model asked for JSON answers with JSON wrapped in a code fence, or with a
sentence in front of it, or with a trailing comma, or with ``//`` comments
explaining the fields, or with the first two thirds of a long object because it
hit the token limit. ``json.loads`` rejects every one of those, and the reply
has already been paid for.

    >>> from llm_json import loads
    >>> loads('Sure! ```json\\n{"ok": true,}\\n```')
    {'ok': True}

The part that is not like other repair libraries: nothing is fixed quietly.

    >>> from llm_json import extract
    >>> found = extract('{"items": [1, 2, 3], "total":')
    >>> found.value
    {'items': [1, 2, 3]}
    >>> found.truncated
    True

``total`` is missing because the model never sent it, and a library that
returns the dict without saying so hands you a record with a silently absent
field. :attr:`~llm_json.extract.Extraction.repairs` lists what was needed, and
``loads(..., allow_truncated=False)`` refuses the partial value outright.
"""

from __future__ import annotations

from llm_json.errors import (
    JSONNotFound,
    LLMJSONError,
    TruncatedJSON,
    UnexpectedType,
)
from llm_json.extract import Extraction, extract, extract_all, loads
from llm_json.repairs import (
    COMMENTS,
    FENCED,
    PROSE,
    PY_LITERALS,
    QUOTES,
    TRAILING_COMMA,
    TRUNCATION,
)

__version__ = "0.1.0"

__all__ = [
    "COMMENTS",
    "FENCED",
    "PROSE",
    "PY_LITERALS",
    "QUOTES",
    "TRAILING_COMMA",
    "TRUNCATION",
    "Extraction",
    "JSONNotFound",
    "LLMJSONError",
    "TruncatedJSON",
    "UnexpectedType",
    "__version__",
    "extract",
    "extract_all",
    "loads",
]
