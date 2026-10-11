"""Run against the replies that motivated the library. No API key needed.

    python examples/demo.py

Every string below is the shape of a real model response. The point of the
output is the right-hand column: what had to be repaired, which is the part a
library that "just works" never tells you.
"""

from __future__ import annotations

from llm_json import JSONNotFound, extract, loads
from llm_json.errors import TruncatedJSON

CASES: list[tuple[str, str]] = [
    (
        "already valid",
        '{"status": "ok"}',
    ),
    (
        "fenced, with a preamble",
        'Sure! Here is the JSON:\n\n```json\n{"status": "ok"}\n```',
    ),
    (
        "trailing comma",
        '{"itens": [1, 2,], "total": 3,}',
    ),
    (
        "comments explaining the fields",
        '{\n  // digits only\n  "cpf": "52998224725",\n'
        '  "site": "https://exemplo.com.br/a//b"\n}',
    ),
    (
        "answered in Python",
        "{'nome': 'Maria', 'ativo': True, 'apelido': None}",
    ),
    (
        "typeset with curly quotes",
        "{“status”: “ok”}",
    ),
    (
        "cut off mid-object",
        '{"cliente": "ACME", "cnpj": "12345678000195", "total":',
    ),
    (
        "cut off mid-record, in a list",
        '[{"id": 1, "ok": true}, {"id": 2, "ok": true}, {"id": 3, "o',
    ),
    (
        "braces in the prose, JSON after",
        'Use {campo} for placeholders. Result: {"status": "ok"}',
    ),
    (
        "a refusal",
        "I am not able to produce that.",
    ),
    (
        "cut off before anything finished",
        '{"cliente": "AC',
    ),
]


def main() -> None:
    print(f"{'case':34} {'value':44} repairs")
    print("-" * 110)

    for label, reply in CASES:
        try:
            found = extract(reply)
        except JSONNotFound:
            print(f"{label:34} {'JSONNotFound':44} -")
            continue
        repairs = ", ".join(found.repairs) or "none"
        print(f"{label:34} {str(found.value)[:43]:44} {repairs}")

    print()
    print("Why the repair list is worth reading:")
    cut_off = '{"registros": [{"id": 1}, {"id": 2}, {"id'
    value = loads(cut_off)
    print(f"  loads(...)                  -> {value}")
    print("  two records, not three: the third one never finished arriving.")

    try:
        loads(cut_off, allow_truncated=False)
    except TruncatedJSON as exc:
        print(f"  allow_truncated=False       -> {type(exc).__name__}")
        print(f"  ...with the partial value   -> {exc.value}")
        print("  so the caller decides: retry with more tokens, or keep these two.")


if __name__ == "__main__":
    main()
