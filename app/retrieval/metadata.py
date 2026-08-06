from typing import Any


Metadata = dict[str, Any]


def string_list(metadata: Metadata, field: str) -> list[str]:
    values = metadata.get(field)
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def highlight_fields(metadata: Metadata) -> dict[str, list[str]]:
    highlight = metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}
    return {
        field: [str(value) for value in highlight[field]]
        for field in ("summary", "keywords")
        if isinstance(highlight.get(field), list)
    }


def merge_metadata(preferred: Metadata, fallback: Metadata) -> Metadata:
    """Merge retrieval evidence while keeping preferred scalar values."""

    merged = {**fallback, **preferred}
    _merge_query_provenance(merged, preferred, fallback)
    _merge_string_list_field(merged, "matched_keywords", preferred, fallback)
    _merge_highlights(merged, preferred, fallback)
    return merged


def _merge_query_provenance(
    merged: Metadata,
    preferred: Metadata,
    fallback: Metadata,
) -> None:
    by_index: dict[int, str] = {}
    unindexed_queries: list[str] = []
    for source in (preferred, fallback):
        queries = source.get("matched_queries")
        indexes = source.get("matched_query_indexes")
        if isinstance(queries, list) and isinstance(indexes, list):
            for raw_index, raw_query in zip(indexes, queries):
                if isinstance(raw_index, int):
                    by_index.setdefault(raw_index, str(raw_query))
        elif isinstance(queries, list):
            for raw_query in queries:
                query = str(raw_query)
                if query not in unindexed_queries:
                    unindexed_queries.append(query)

    if by_index:
        ordered = sorted(by_index.items())
        merged["matched_query_indexes"] = [index for index, _ in ordered]
        merged["matched_queries"] = [query for _, query in ordered]
    elif unindexed_queries:
        merged["matched_queries"] = unindexed_queries


def _merge_string_list_field(
    merged: Metadata,
    field: str,
    preferred: Metadata,
    fallback: Metadata,
) -> None:
    values: list[str] = []
    seen: set[str] = set()
    for source in (preferred, fallback):
        for value in string_list(source, field):
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
    if values:
        merged[field] = values


def _merge_highlights(
    merged: Metadata,
    preferred: Metadata,
    fallback: Metadata,
) -> None:
    preferred_highlight = highlight_fields(preferred)
    fallback_highlight = highlight_fields(fallback)
    if not preferred_highlight and not fallback_highlight:
        return

    highlight: dict[str, list[str]] = {}
    for field in ("summary", "keywords"):
        values: list[str] = []
        seen: set[str] = set()
        for source in (preferred_highlight, fallback_highlight):
            for value in source.get(field, []):
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)
        if values:
            highlight[field] = values
    if highlight:
        merged["highlight"] = highlight
