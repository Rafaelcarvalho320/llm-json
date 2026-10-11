"""The pipeline, the reporting, and the knobs a caller gets."""

from __future__ import annotations

import dataclasses

import pytest

from llm_json import (
    COMMENTS,
    FENCED,
    PROSE,
    PY_LITERALS,
    QUOTES,
    TRAILING_COMMA,
    TRUNCATION,
    Extraction,
    JSONNotFound,
    TruncatedJSON,
    UnexpectedType,
    extract,
    extract_all,
    loads,
)


class TestCleanInput:
    def test_object(self) -> None:
        found = extract('{"a": 1}')
        assert found.value == {"a": 1}
        assert found.repairs == ()
        assert found.clean
        assert not found.truncated

    def test_array(self) -> None:
        assert extract("[1, 2]").value == [1, 2]

    def test_surrounding_whitespace_is_not_a_repair(self) -> None:
        assert extract('\n  {"a": 1}\n  ').clean

    def test_unicode_survives(self) -> None:
        assert extract('{"nome": "João"}').value == {"nome": "João"}

    def test_escaped_sequences_survive(self) -> None:
        assert extract('{"a": "line\\nbreak"}').value == {"a": "line\nbreak"}


class TestRepairsAreReported:
    def test_fence(self) -> None:
        found = extract('```json\n{"a": 1}\n```')
        assert found.value == {"a": 1}
        assert found.repairs == (FENCED,)

    def test_prose(self) -> None:
        found = extract('Here you go: {"a": 1}')
        assert found.repairs == (PROSE,)

    def test_trailing_comma(self) -> None:
        found = extract('{"a": 1,}')
        assert found.repairs == (TRAILING_COMMA,)

    def test_comments(self) -> None:
        found = extract('{"a": 1 // the one\n}')
        assert found.repairs == (COMMENTS,)

    def test_python_literals(self) -> None:
        found = extract('{"a": True}')
        assert found.repairs == (PY_LITERALS,)

    def test_quotes(self) -> None:
        found = extract("{'a': 1}")
        assert found.repairs == (QUOTES,)

    def test_truncation(self) -> None:
        found = extract('{"a": "x"')
        assert found.repairs == (TRUNCATION,)
        assert found.truncated

    def test_several_repairs_in_application_order(self) -> None:
        found = extract('Sure:\n```json\n{\n  // id\n  "a": 1,\n}\n```')
        assert found.value == {"a": 1}
        assert found.repairs == (FENCED, COMMENTS, TRAILING_COMMA)

    def test_only_necessary_repairs_are_recorded(self) -> None:
        # A merely fenced response must not also be reported as quote-fixed.
        found = extract('```\n{"a": 1}\n```')
        assert found.repairs == (FENCED,)

    def test_repaired_text_is_available_for_logging(self) -> None:
        found = extract('{"a": 1,}')
        assert found.text == '{"a": 1}'


class TestSpan:
    def test_span_points_into_the_original_text(self) -> None:
        text = 'blah {"a": 1} blah'
        found = extract(text)
        assert found.span is not None
        start, end = found.span
        assert text[start:end] == '{"a": 1}'

    def test_span_is_none_for_a_fenced_block(self) -> None:
        # The offsets would refer to the block, not the response, and a span
        # that silently means something else is worse than no span.
        assert extract('```json\n{"a": 1}\n```').span is None


class TestCandidateOrder:
    def test_a_fenced_block_wins_over_prose_braces(self) -> None:
        text = 'Use {placeholder}:\n```json\n{"a": 1}\n```'
        assert extract(text).value == {"a": 1}

    def test_balanced_prose_does_not_shadow_the_real_object(self) -> None:
        # {placeholder} is balanced and fails to parse; the bug worth a test
        # is it being "closed" into a valid {} that outranks the real answer.
        assert extract('Use {placeholder} syntax. Result: {"a": 1}').value == {"a": 1}

    def test_first_of_several_objects(self) -> None:
        assert extract('{"a": 1} and {"b": 2}').value == {"a": 1}


class TestExtractAll:
    def test_two_objects(self) -> None:
        values = [found.value for found in extract_all('{"a": 1}\n{"b": 2}')]
        assert values == [{"a": 1}, {"b": 2}]

    def test_two_fenced_blocks(self) -> None:
        text = '```json\n{"a": 1}\n```\nand\n```json\n{"b": 2}\n```'
        assert [f.value for f in extract_all(text)] == [{"a": 1}, {"b": 2}]

    def test_duplicate_text_is_not_yielded_twice(self) -> None:
        # The fenced body and the whole-text scan see the same object.
        found = list(extract_all('```json\n{"a": 1}\n```'))
        assert len(found) == 1

    def test_no_json_yields_nothing(self) -> None:
        assert list(extract_all("just prose")) == []

    def test_is_lazy(self) -> None:
        iterator = extract_all('{"a": 1} {"b": 2}')
        assert next(iterator).value == {"a": 1}


class TestNotFound:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "no json at all",
            "I cannot help with that.",
            "{",
            '{"a"',
            "{unquoted: 1}",
        ],
    )
    def test_raises(self, text: str) -> None:
        with pytest.raises(JSONNotFound):
            extract(text)

    def test_the_offending_text_is_on_the_exception(self) -> None:
        with pytest.raises(JSONNotFound) as info:
            extract("sorry, no")
        assert info.value.text == "sorry, no"

    def test_long_text_is_truncated_in_the_message(self) -> None:
        with pytest.raises(JSONNotFound) as info:
            extract("x" * 5000)
        assert len(str(info.value)) < 300
        assert str(info.value).endswith("...'")

    def test_it_is_a_value_error(self) -> None:
        # So code that already catches json.loads keeps working.
        with pytest.raises(ValueError):
            extract("nope")


class TestLoads:
    def test_matches_json_loads_on_clean_input(self) -> None:
        assert loads('{"a": [1, null, true]}') == {"a": [1, None, True]}

    def test_repairs_silently_when_asked_to(self) -> None:
        assert loads('```json\n{"a": 1,}\n```') == {"a": 1}

    def test_truncated_is_allowed_by_default(self) -> None:
        assert loads('{"a": "x"') == {"a": "x"}

    def test_truncated_can_be_refused(self) -> None:
        with pytest.raises(TruncatedJSON) as info:
            loads('{"a": "x"', allow_truncated=False)
        # The partial value rides along: deciding between "retry" and "use
        # what arrived" needs it in hand.
        assert info.value.value == {"a": "x"}

    def test_refusing_truncation_does_not_reject_other_repairs(self) -> None:
        assert loads('{"a": 1,}', allow_truncated=False) == {"a": 1}

    def test_expect_dict_passes(self) -> None:
        assert loads('{"a": 1}', expect=dict) == {"a": 1}

    def test_expect_dict_rejects_a_list(self) -> None:
        # The classic prompt failure: the schema said object, the model
        # wrapped it in a list of one.
        with pytest.raises(UnexpectedType) as info:
            loads('[{"a": 1}]', expect=dict)
        assert info.value.value == [{"a": 1}]

    def test_expect_accepts_a_tuple(self) -> None:
        assert loads("[1]", expect=(dict, list)) == [1]

    def test_expect_message_names_both_types(self) -> None:
        with pytest.raises(UnexpectedType) as info:
            loads("[1]", expect=(dict, str))
        assert "dict or str" in str(info.value)

    def test_bool_does_not_satisfy_int(self) -> None:
        # bool is a subclass of int, so a plain isinstance check lets `true`
        # through as a number. That trap is closed deliberately.
        with pytest.raises(UnexpectedType):
            loads("true", expect=int)

    def test_bool_against_bool(self) -> None:
        assert loads("true", expect=bool) is True


class TestExtractionDataclass:
    def test_is_frozen(self) -> None:
        found = extract('{"a": 1}')
        with pytest.raises(dataclasses.FrozenInstanceError):
            found.value = 2  # type: ignore[misc]

    def test_can_be_constructed_directly(self) -> None:
        found = Extraction(value={}, text="{}", span=None)
        assert found.clean
        assert not found.truncated
