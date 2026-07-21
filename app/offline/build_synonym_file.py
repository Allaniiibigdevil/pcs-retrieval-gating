import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_SYNONYM_SEPARATOR_RE = re.compile(r"[,，]")


@dataclass(frozen=True)
class SynonymBuildResult:
    source_entries: int
    rules_written: int
    synonyms_written: int
    skipped_entries: int
    merged_entries: int
    truncated_synonyms: int


def _normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return _WHITESPACE_RE.sub(" ", normalized)


def _parse_synonyms(value: object) -> list[str]:
    if isinstance(value, str):
        values = _SYNONYM_SEPARATOR_RE.split(value)
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        return []
    return [term for item in values if (term := _normalize_term(item))]


def _escape_solr_term(term: str) -> str:
    if "=>" in term:
        raise ValueError(f"Synonym term cannot contain '=>': {term!r}")
    return term.replace("\\", "\\\\").replace(",", "\\,")


def build_directional_synonym_file(
    input_path: str | Path,
    output_path: str | Path,
    max_synonyms_per_rule: int | None = None,
) -> SynonymBuildResult:
    if max_synonyms_per_rule is not None and max_synonyms_per_rule < 1:
        raise ValueError("max_synonyms_per_rule must be at least 1")

    source_path = Path(input_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Synonym JSON must have the shape {'data': [...]}")

    entries = payload["data"]
    synonyms_by_word: dict[str, list[str]] = {}
    seen_by_word: dict[str, set[str]] = {}
    skipped_entries = 0
    merged_entries = 0

    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("word"), str):
            skipped_entries += 1
            continue

        word = _normalize_term(entry["word"])
        synonyms = _parse_synonyms(entry.get("synonyms"))
        if not word or not synonyms:
            skipped_entries += 1
            continue

        _escape_solr_term(word)
        for synonym in synonyms:
            _escape_solr_term(synonym)

        if word in synonyms_by_word:
            merged_entries += 1
        else:
            synonyms_by_word[word] = []
            seen_by_word[word] = {word}

        for synonym in synonyms:
            if synonym in seen_by_word[word]:
                continue
            seen_by_word[word].add(synonym)
            synonyms_by_word[word].append(synonym)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.name}.tmp")
    rules_written = 0
    synonyms_written = 0
    truncated_synonyms = 0

    with temporary_output.open("w", encoding="utf-8", newline="\n") as file:
        for word, synonyms in synonyms_by_word.items():
            if not synonyms:
                skipped_entries += 1
                continue
            if max_synonyms_per_rule is not None and len(synonyms) > max_synonyms_per_rule:
                truncated_synonyms += len(synonyms) - max_synonyms_per_rule
                synonyms = synonyms[:max_synonyms_per_rule]

            left = _escape_solr_term(word)
            right = ", ".join(_escape_solr_term(term) for term in [word, *synonyms])
            file.write(f"{left} => {right}\n")
            rules_written += 1
            synonyms_written += len(synonyms)

    temporary_output.replace(output)
    return SynonymBuildResult(
        source_entries=len(entries),
        rules_written=rules_written,
        synonyms_written=synonyms_written,
        skipped_entries=skipped_entries,
        merged_entries=merged_entries,
        truncated_synonyms=truncated_synonyms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a JSON synonym dictionary into one-way Elasticsearch rules."
    )
    parser.add_argument("--input", required=True, help="Input JSON dictionary path.")
    parser.add_argument("--output", required=True, help="Output Elasticsearch synonym file path.")
    parser.add_argument(
        "--max-synonyms-per-rule",
        type=int,
        default=None,
        help="Optional cap for synonyms on the right side of each rule.",
    )
    args = parser.parse_args()

    result = build_directional_synonym_file(
        args.input,
        args.output,
        max_synonyms_per_rule=args.max_synonyms_per_rule,
    )
    print(
        "built_synonym_file "
        f"source_entries={result.source_entries} "
        f"rules_written={result.rules_written} "
        f"synonyms_written={result.synonyms_written} "
        f"skipped_entries={result.skipped_entries} "
        f"merged_entries={result.merged_entries} "
        f"truncated_synonyms={result.truncated_synonyms} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
