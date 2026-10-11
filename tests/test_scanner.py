"""The single pass everything else is built on."""

from __future__ import annotations

import pytest

from llm_json.scanner import (
    CODE,
    COMMENT,
    STRING,
    candidate_spans,
    classify,
    walk,
)


def kinds_of(text: str) -> str:
    """Render :func:`classify` readably: ``.`` code, ``s`` string, ``c`` comment."""
    symbols = {CODE: ".", STRING: "s", COMMENT: "c"}
    return "".join(symbols[kind] for kind in classify(text))


class TestClassify:
    def test_plain_object(self) -> None:
        assert kinds_of('{"a":1}') == ".sss..."

    def test_quotes_belong_to_the_string(self) -> None:
        # The delimiters are labelled as string too, so a caller can rebuild
        # the text by filtering on labels alone.
        assert kinds_of('"x"') == "sss"

    def test_escaped_quote_does_not_end_the_string(self) -> None:
        assert kinds_of(r'"a\"b"') == "ssssss"

    def test_escaped_backslash_then_quote_does_end_it(self) -> None:
        text = r'"a\\" '
        assert kinds_of(text) == "sssss."

    def test_line_comment(self) -> None:
        assert kinds_of("1 // x\n2") == "..cccc.."

    def test_line_comment_stops_at_the_newline(self) -> None:
        assert classify("// x\n1")[-1] == CODE

    def test_block_comment(self) -> None:
        assert kinds_of("1/* x */2") == ".ccccccc."

    def test_unterminated_block_comment_runs_to_the_end(self) -> None:
        assert set(classify("/* nope")) == {COMMENT}

    def test_unterminated_string_runs_to_the_end(self) -> None:
        assert set(classify('"nope')) == {STRING}

    def test_slashes_inside_a_string_are_not_a_comment(self) -> None:
        # The bug this whole module exists to prevent.
        text = '{"url": "https://example.com"}'
        assert COMMENT not in classify(text)

    def test_quote_inside_a_comment_does_not_open_a_string(self) -> None:
        text = '// he said "hi\n{"a": 1}'
        kinds = classify(text)
        assert kinds[text.index('"hi')] == COMMENT
        assert kinds[text.index("{")] == CODE

    def test_empty_text(self) -> None:
        assert classify("") == []


class TestCandidateSpans:
    def test_single_object(self) -> None:
        assert list(candidate_spans('{"a":1}')) == [(0, 7)]

    def test_prose_before_and_after(self) -> None:
        text = 'ok: {"a":1} done'
        [(start, end)] = list(candidate_spans(text))
        assert text[start:end] == '{"a":1}'

    def test_two_objects_are_two_candidates(self) -> None:
        text = '{"a":1} and {"b":2}'
        spans = list(candidate_spans(text))
        assert [text[s:e] for s, e in spans] == ['{"a":1}', '{"b":2}']

    def test_nested_braces_do_not_produce_extra_candidates(self) -> None:
        text = '{"a": {"b": 1}}'
        assert list(candidate_spans(text)) == [(0, len(text))]

    def test_brace_inside_a_string_is_not_an_opener(self) -> None:
        text = '{"a": "{"}'
        assert list(candidate_spans(text)) == [(0, len(text))]

    def test_brace_inside_a_comment_is_ignored(self) -> None:
        text = '// { not json\n{"a":1}'
        [(start, end)] = list(candidate_spans(text))
        assert text[start:end] == '{"a":1}'

    def test_truncated_span_ends_at_the_end_of_the_text(self) -> None:
        text = '{"a": [1,'
        assert list(candidate_spans(text)) == [(0, len(text))]

    def test_array_is_a_candidate(self) -> None:
        assert list(candidate_spans("[1,2]")) == [(0, 5)]

    def test_no_json_no_candidates(self) -> None:
        assert list(candidate_spans("nothing here")) == []


class TestWalkComplete:
    @pytest.mark.parametrize(
        "text",
        [
            "{}",
            "[]",
            '{"a": 1}',
            "[1, 2, 3]",
            '{"a": {"b": [1, {"c": null}]}}',
            '{"a": "}"}',
            '{"a": "\\""}',
        ],
    )
    def test_balanced_text_is_complete(self, text: str) -> None:
        assert walk(text).complete

    def test_complete_text_is_returned_untouched(self) -> None:
        text = '{"a": 1}'
        assert walk(text).closed(text) == text


class TestWalkTruncation:
    def test_missing_closing_brace(self) -> None:
        text = '{"a": "x"'
        scan = walk(text)
        assert not scan.complete
        assert scan.closed(text) == '{"a": "x"}'

    def test_nested_containers_close_innermost_first(self) -> None:
        text = '{"a": {"b": [true'
        assert walk(text).closed(text) == '{"a": {"b": [true]}}'

    def test_dangling_key_is_dropped(self) -> None:
        # Closing after the key would give {"a": 1, "b"} which is not JSON.
        text = '{"a": 1, "b":'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_key_without_a_colon_is_dropped(self) -> None:
        text = '{"a": 1, "b"'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_string_value_is_kept(self) -> None:
        text = '{"a": "done", "b":'
        assert walk(text).closed(text) == '{"a": "done"}'

    def test_unterminated_string_is_dropped(self) -> None:
        text = '{"a": 1, "b": "half'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_string_ending_in_an_escaped_quote_is_still_unterminated(self) -> None:
        # The naive "does it end with a quote" test reads this as closed.
        text = '{"a": 1, "b": "c\\"'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_number_against_the_end_is_dropped(self) -> None:
        # 12 parses, but it is also a prefix of 125.
        text = '{"a": 1, "b": 12'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_number_followed_by_a_comma_is_kept(self) -> None:
        text = '{"a": 1, "b": 12,'
        assert walk(text).closed(text) == '{"a": 1, "b": 12}'

    def test_keyword_against_the_end_is_kept(self) -> None:
        # Nothing valid starts with "true" and continues.
        text = '{"a": true'
        assert walk(text).closed(text) == '{"a": true}'

    def test_partial_keyword_is_dropped(self) -> None:
        text = '{"a": 1, "b": tru'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_array_elements_before_the_cut_are_kept(self) -> None:
        text = '[{"id": 1}, {"id": 2}, {"id":'
        assert walk(text).closed(text) == '[{"id": 1}, {"id": 2}]'

    def test_opening_a_container_is_not_a_safe_point(self) -> None:
        # Marking after the opener yields [1, 2, {}]: a record nobody sent.
        text = "[1, 2, {"
        assert walk(text).closed(text) == "[1, 2]"

    def test_empty_container_as_a_value_survives(self) -> None:
        text = '{"a": [], "b":'
        assert walk(text).closed(text) == '{"a": []}'

    def test_comment_before_the_cut_does_not_confuse_the_walk(self) -> None:
        text = '{"a": 1, // note\n"b":'
        assert walk(text).closed(text) == '{"a": 1}'


class TestWalkSalvageable:
    @pytest.mark.parametrize("text", ["{", "[", '{"a"', '{"a":', '{"ab', "{ "])
    def test_nothing_complete_means_nothing_to_salvage(self, text: str) -> None:
        assert not walk(text).salvageable

    @pytest.mark.parametrize("text", ['{"a": "x"', "[true", '{"a": {}'])
    def test_one_complete_value_is_salvageable(self, text: str) -> None:
        assert walk(text).salvageable

    @pytest.mark.parametrize("text", ['{"a": 1', "[12"])
    def test_a_lone_trailing_number_is_not_salvage(self, text: str) -> None:
        """``{"a": 1`` recovers nothing, and that is the intended answer.

        The only value present touches the end of the input, so it may be the
        first digit of a longer number. Guessing it writes a wrong number into
        somebody's database; refusing tells the caller to ask again.
        """
        assert not walk(text).salvageable

    def test_balanced_prose_is_not_salvageable(self) -> None:
        # {placeholder} is balanced, fails to parse, and must not be rescued
        # into a valid empty object.
        assert not walk("{placeholder}").salvageable


class TestObjectArrayContract:
    """An object keeps the members that arrived; an array keeps whole elements."""

    def test_an_object_keeps_the_members_that_arrived(self) -> None:
        text = '{"a": "x", "b": "y", "c":'
        assert walk(text).closed(text) == '{"a": "x", "b": "y"}'

    def test_an_array_drops_an_unfinished_element(self) -> None:
        text = '[{"id": 1}, {"id": 2}, {"id": 3, "status":'
        assert walk(text).closed(text) == '[{"id": 1}, {"id": 2}]'

    def test_an_array_drops_an_unfinished_nested_array(self) -> None:
        text = "[[1, 2], [3,"
        assert walk(text).closed(text) == "[[1, 2]]"

    def test_the_outermost_unfinished_array_decides(self) -> None:
        text = '{"rows": [{"id": 1}, {"id": 2, "cells": [9,'
        assert walk(text).closed(text) == '{"rows": [{"id": 1}]}'

    def test_a_nested_object_keeps_its_own_members(self) -> None:
        # Not an array element, so the partial inner object is still worth
        # having: the fields inside it did arrive.
        text = '{"a": {"b": "x", "c":'
        assert walk(text).closed(text) == '{"a": {"b": "x"}}'

    def test_it_backs_out_to_an_ancestor_that_has_something(self) -> None:
        # The innermost frame is empty; cutting there would salvage nothing
        # even though {"a": 1} was right there.
        text = '{"a": 1, "b": {"c":'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_it_backs_out_through_two_empty_levels(self) -> None:
        text = '{"a": 1, "b": {"c": {'
        assert walk(text).closed(text) == '{"a": 1}'

    def test_nothing_anywhere_is_not_salvageable(self) -> None:
        assert not walk('{"a": {"b":').salvageable
