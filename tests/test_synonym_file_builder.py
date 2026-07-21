import json

import pytest

from app.offline.build_synonym_file import build_directional_synonym_file


def test_build_directional_synonym_file_preserves_source_and_merges_duplicates(tmp_path) -> None:
    input_path = tmp_path / "synonyms.json"
    output_path = tmp_path / "pcs_synonyms.txt"
    input_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "word": "美国",
                        "synonyms": "美利坚合众国,United States,united states,USA",
                    },
                    {"word": " 美国 ", "synonyms": "花旗国，USA"},
                    {"word": "还了", "synonyms": "还"},
                    {"word": "", "synonyms": "无效"},
                    {"word": "空", "synonyms": ""},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_directional_synonym_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "美国 => 美国, 美利坚合众国, united states, usa, 花旗国",
        "还了 => 还了, 还",
    ]
    assert result.source_entries == 5
    assert result.rules_written == 2
    assert result.synonyms_written == 5
    assert result.skipped_entries == 2
    assert result.merged_entries == 1
    assert result.truncated_synonyms == 0


def test_build_directional_synonym_file_can_cap_large_groups(tmp_path) -> None:
    input_path = tmp_path / "synonyms.json"
    output_path = tmp_path / "pcs_synonyms.txt"
    input_path.write_text(
        json.dumps({"data": [{"word": "美国", "synonyms": "美利坚,花旗国,usa"}]}),
        encoding="utf-8",
    )

    result = build_directional_synonym_file(
        input_path,
        output_path,
        max_synonyms_per_rule=2,
    )

    assert output_path.read_text(encoding="utf-8") == "美国 => 美国, 美利坚, 花旗国\n"
    assert result.truncated_synonyms == 1


def test_build_directional_synonym_file_rejects_invalid_shape(tmp_path) -> None:
    input_path = tmp_path / "synonyms.json"
    input_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must have the shape"):
        build_directional_synonym_file(input_path, tmp_path / "pcs_synonyms.txt")
