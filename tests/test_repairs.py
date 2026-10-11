"""Each repair on its own, including the cases its regex version gets wrong."""

from __future__ import annotations

import pytest

from llm_json.repairs import (
    close_truncation,
    drop_trailing_commas,
    fenced_blocks,
    normalise_quotes,
    python_literals,
    strip_comments,
)


class TestFencedBlocks:
    def test_plain_fence(self) -> None:
        assert fenced_blocks("```\n{}\n```") == ["{}\n"]

    def test_language_tag_is_consumed(self) -> None:
        assert fenced_blocks('```json\n{"a":1}\n```') == ['{"a":1}\n']

    @pytest.mark.parametrize(
        "tag", ["json", "JSON", "Json", "json5", "javascript", "js"]
    )
    def test_tags_models_actually_emit(self, tag: str) -> None:
        assert fenced_blocks(f"```{tag}\n{{}}\n```") == ["{}\n"]

    def test_unclosed_fence_still_yields_its_body(self) -> None:
        # The ordinary shape of a truncated reply: the fence opened, the
        # object started, the tokens ran out. Requiring the closing fence
        # throws away the only content there is.
        assert fenced_blocks('```json\n{"a": 1,') == ['{"a": 1,']

    def test_two_blocks(self) -> None:
        text = "first\n```\n{}\n```\nthen\n```\n[]\n```"
        assert fenced_blocks(text) == ["{}\n", "[]\n"]

    def test_no_fence(self) -> None:
        assert fenced_blocks('{"a":1}') == []

    def test_prose_outside_the_fence_is_not_included(self) -> None:
        assert fenced_blocks('Sure!\n```json\n{"a":1}\n```\nEnjoy.') == ['{"a":1}\n']


class TestStripComments:
    def test_line_comment(self) -> None:
        assert strip_comments('{"a": 1} // done') == '{"a": 1} '

    def test_block_comment(self) -> None:
        assert strip_comments('{/* note */"a": 1}') == '{"a": 1}'

    def test_comment_between_fields(self) -> None:
        assert strip_comments('{\n// id\n"a": 1\n}') == '{\n\n"a": 1\n}'

    def test_url_in_a_string_survives(self) -> None:
        # The whole reason this is not a regex. A regex stripper turns this
        # into {"url": "https:} - valid-looking and silently wrong.
        text = '{"url": "https://example.com/a//b"}'
        assert strip_comments(text) == text

    def test_block_comment_markers_inside_a_string_survive(self) -> None:
        text = '{"glob": "/*.py"}'
        assert strip_comments(text) == text

    def test_text_without_comments_is_returned_unchanged(self) -> None:
        text = '{"a": 1}'
        assert strip_comments(text) is text

    def test_newline_is_kept_so_following_lines_remain_separate(self) -> None:
        assert strip_comments("1 // x\n2") == "1 \n2"


class TestDropTrailingCommas:
    def test_object(self) -> None:
        assert drop_trailing_commas('{"a": 1,}') == '{"a": 1}'

    def test_array(self) -> None:
        assert drop_trailing_commas("[1, 2,]") == "[1, 2]"

    def test_with_whitespace_and_newlines(self) -> None:
        assert drop_trailing_commas('{"a": 1,\n  }') == '{"a": 1\n  }'

    def test_nested(self) -> None:
        assert drop_trailing_commas('{"a": [1,], "b": 2,}') == '{"a": [1], "b": 2}'

    def test_comma_inside_a_string_survives(self) -> None:
        # `,\s*}` matches inside this string and corrupts the value.
        text = '{"note": "a, } b"}'
        assert drop_trailing_commas(text) == text

    def test_separating_comma_is_untouched(self) -> None:
        text = '{"a": 1, "b": 2}'
        assert drop_trailing_commas(text) == text

    def test_text_without_trailing_commas_is_returned_unchanged(self) -> None:
        text = "[1, 2]"
        assert drop_trailing_commas(text) is text


class TestNormaliseQuotes:
    def test_curly_doubles(self) -> None:
        assert normalise_quotes("{“a”: “b”}") == '{"a": "b"}'

    def test_single_quoted_object(self) -> None:
        assert normalise_quotes("{'a': 'b'}") == '{"a": "b"}'

    def test_guard_leaves_proper_json_alone(self) -> None:
        # A curly quote inside a properly delimited string is somebody's
        # words, and rewriting it corrupts the value while fixing nothing.
        text = '{"said": “hello”}'
        assert normalise_quotes(text) is text

    def test_apostrophe_in_a_proper_string_survives(self) -> None:
        text = '{"note": "don\'t"}'
        assert normalise_quotes(text) is text

    def test_curly_singles_when_there_is_no_double_quote_at_all(self) -> None:
        assert normalise_quotes("{‘a’: ‘b’}") == '{"a": "b"}'


class TestPythonLiterals:
    def test_true_false_none(self) -> None:
        got = python_literals('{"a": True, "b": False, "c": None}')
        assert got == '{"a": true, "b": false, "c": null}'

    def test_word_inside_a_string_survives(self) -> None:
        text = '{"note": "True story", "x": "None of it"}'
        assert python_literals(text) == text

    def test_only_whole_words(self) -> None:
        text = '{"Truest": 1}'
        assert python_literals(text) == text

    def test_text_without_python_literals_is_returned_unchanged(self) -> None:
        text = '{"a": true}'
        assert python_literals(text) is text


class TestCloseTruncation:
    def test_closes_what_is_open(self) -> None:
        assert close_truncation('{"a": "x"') == '{"a": "x"}'

    def test_complete_text_is_untouched(self) -> None:
        text = '{"a": 1}'
        assert close_truncation(text) is text

    def test_refuses_when_there_is_nothing_to_salvage(self) -> None:
        # Returning {} here would read downstream as "the model found
        # nothing" rather than "the model was interrupted".
        assert close_truncation('{"a"') == '{"a"'

    def test_refuses_on_balanced_prose(self) -> None:
        assert close_truncation("{placeholder}") == "{placeholder}"

    def test_list_of_records_keeps_the_complete_ones(self) -> None:
        got = close_truncation('[{"id": 1}, {"id": 2}, {"id"')
        assert got == '[{"id": 1}, {"id": 2}]'
