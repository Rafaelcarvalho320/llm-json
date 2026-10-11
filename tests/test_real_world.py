"""Whole replies, in the shapes models actually send them.

The unit tests above each poke at one repair. These are the end-to-end cases
that motivated the package, kept verbatim so a refactor that breaks a real
response fails here rather than in somebody's integration.
"""

from __future__ import annotations

from llm_json import TRUNCATION, extract, loads

POLITE = """Of course! Here is the JSON you asked for:

```json
{
  "cliente": "ACME Ltda",
  "cnpj": "12.345.678/0001-95",
  "itens": [
    {"sku": "A-1", "qtd": 2},
    {"sku": "B-7", "qtd": 1}
  ],
  "total": 149.9
}
```

Let me know if you need anything else!"""

COMMENTED = """{
  // the document number, digits only
  "documento": "12345678901",
  "endereco": "https://maps.example.com/?q=rua//alta",
  /* the value is in cents to avoid float drift */
  "valor_centavos": 14990,
}"""

PYTHONIC = """{'nome': 'Maria', 'ativo': True, 'excluido': False, 'apelido': None}"""

CUT_OFF = """```json
{
  "registros": [
    {"id": 1, "status": "ok"},
    {"id": 2, "status": "ok"},
    {"id": 3, "stat"""

REFUSAL = "I am not able to produce that. Let me know if I can help another way."

NO_FENCE_BUT_PROSE = 'The result is {"ok": true} and the confidence is high.'


class TestPoliteReply:
    def test_value(self) -> None:
        value = loads(POLITE)
        assert value["cliente"] == "ACME Ltda"
        assert value["total"] == 149.9
        assert len(value["itens"]) == 2

    def test_it_was_not_clean(self) -> None:
        assert not extract(POLITE).clean

    def test_but_it_was_not_truncated_either(self) -> None:
        assert not extract(POLITE).truncated


class TestCommentedReply:
    def test_value(self) -> None:
        value = loads(COMMENTED)
        assert value["documento"] == "12345678901"
        assert value["valor_centavos"] == 14990

    def test_the_url_survived_the_comment_stripper(self) -> None:
        # Two slashes inside a string, in the same document as two real
        # comments. This is the case a regex gets wrong.
        assert loads(COMMENTED)["endereco"] == "https://maps.example.com/?q=rua//alta"


class TestPythonicReply:
    def test_value(self) -> None:
        assert loads(PYTHONIC) == {
            "nome": "Maria",
            "ativo": True,
            "excluido": False,
            "apelido": None,
        }


class TestCutOffReply:
    def test_the_complete_records_are_recovered(self) -> None:
        value = loads(CUT_OFF)
        assert value == {
            "registros": [
                {"id": 1, "status": "ok"},
                {"id": 2, "status": "ok"},
            ]
        }

    def test_the_half_written_record_is_not_invented(self) -> None:
        # Not three records with an empty third one: two records, and a flag
        # saying the answer was cut off.
        assert len(loads(CUT_OFF)["registros"]) == 2

    def test_it_is_reported_as_truncated(self) -> None:
        found = extract(CUT_OFF)
        assert found.truncated
        assert TRUNCATION in found.repairs


class TestRefusal:
    def test_a_refusal_is_not_json(self) -> None:
        import pytest

        from llm_json import JSONNotFound

        with pytest.raises(JSONNotFound):
            loads(REFUSAL)


class TestProseWithoutAFence:
    def test_value(self) -> None:
        assert loads(NO_FENCE_BUT_PROSE) == {"ok": True}
