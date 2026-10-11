"""Check the installed package, not the working copy.

Run as a script rather than collected by pytest::

    python tests/smoke_installed.py

Two claims are verified here that the ordinary suite cannot make, because the
ordinary suite imports from ``src`` with the dev extras present:

* **Zero dependencies.** The package says so in its metadata, and a stray
  ``import requests`` is the easiest way for that to quietly stop being true.
* **It works as installed.** A missing entry in ``packages`` publishes a wheel
  that imports and then raises ``ModuleNotFoundError`` on first real use.

The exit code is what CI reads, so every check is an assertion.
"""

from __future__ import annotations

import sys


def main() -> int:
    import llm_json

    # Only the standard library and this package may be loaded by importing it.
    allowed = {"llm_json"}
    third_party = {
        name.split(".")[0]
        for name, module in sys.modules.items()
        if not name.startswith("_")
        and getattr(module, "__file__", None)
        and "site-packages" in str(module.__file__)
    }
    unexpected = third_party - allowed
    assert not unexpected, f"llm_json pulled in {sorted(unexpected)}"

    assert llm_json.loads('```json\n{"a": 1,}\n```') == {"a": 1}

    found = llm_json.extract('{"a": [1, 2], "b":')
    assert found.truncated, found
    assert found.value == {"a": [1, 2]}, found

    found = llm_json.extract('{"url": "https://x.test/a//b"} // note')
    assert found.value == {"url": "https://x.test/a//b"}, found

    try:
        llm_json.loads("I cannot help with that.")
    except llm_json.JSONNotFound:
        pass
    else:  # pragma: no cover - the assertion below reports it
        raise AssertionError("prose should not parse as JSON")

    print(f"ok: llm-json {llm_json.__version__} installed, no third-party imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
