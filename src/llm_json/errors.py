"""Failures a caller is expected to handle.

All of them derive from :class:`ValueError`, so code that already catches the
``ValueError`` from :func:`json.loads` keeps working when it switches to this
package.
"""

from __future__ import annotations


class LLMJSONError(ValueError):
    """Base class for everything this package raises."""


class JSONNotFound(LLMJSONError):
    """No JSON value could be recovered from the text.

    The text is carried on the exception so it can be logged or fed back to
    the model, which is the usual next move. It is truncated in ``str()``:
    a model's reply can be tens of kilobytes, and an exception message is
    often written straight to a log line.
    """

    #: How much of the offending text goes into the message.
    PREVIEW = 200

    def __init__(self, text: str) -> None:
        self.text = text
        preview = text[: self.PREVIEW]
        if len(text) > self.PREVIEW:
            preview += "..."
        super().__init__(f"no JSON value found in {preview!r}")


class TruncatedJSON(LLMJSONError):
    """A value was recovered, but only by closing a cut-off response.

    Raised instead of returning, when the caller passed
    ``allow_truncated=False``. The repaired value is on the exception rather
    than discarded: deciding between "retry with a higher token limit" and
    "use the three records that did arrive" belongs to the caller, and that
    decision needs the partial value in hand.
    """

    def __init__(self, value: object, text: str) -> None:
        self.value = value
        self.text = text
        super().__init__(
            "the response was cut off and had to be closed to parse; "
            "pass allow_truncated=True to accept the partial value"
        )


class UnexpectedType(LLMJSONError):
    """The JSON parsed, but is not the shape the caller asked for.

    Worth its own class because it is the most common way a prompt fails in
    production: the schema says object, the model wraps it in a list of one,
    and every downstream ``value["id"]`` raises ``TypeError`` somewhere far
    from the cause.
    """

    def __init__(self, value: object, expected: type | tuple[type, ...]) -> None:
        self.value = value
        self.expected = expected
        names = expected if isinstance(expected, tuple) else (expected,)
        wanted = " or ".join(item.__name__ for item in names)
        super().__init__(f"expected {wanted}, got {type(value).__name__}")
