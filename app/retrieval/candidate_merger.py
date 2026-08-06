from app.schemas.search import SearchHit


class CandidateMerger:
    def merge(
        self,
        source_id: str,
        bm25_hits: list[SearchHit],
        vector_hits: list[SearchHit],
    ) -> list[SearchHit]:
        _validate_source(source_id, bm25_hits)
        _validate_source(source_id, vector_hits)
        merged: dict[str, SearchHit] = {}

        for hit in bm25_hits:
            merged[hit.doc_id] = hit.model_copy(deep=True)

        for hit in vector_hits:
            existing = merged.get(hit.doc_id)
            if existing is None:
                merged[hit.doc_id] = hit.model_copy(deep=True)
                continue

            existing.vector_score = hit.vector_score
            existing.vector_rank = hit.vector_rank
            if existing.summary is None:
                existing.summary = hit.summary
            if not existing.keywords:
                existing.keywords = list(hit.keywords)
            existing.metadata = _merge_metadata(existing.metadata, hit.metadata)

        return list(merged.values())


def _validate_source(source_id: str, hits: list[SearchHit]) -> None:
    mismatched = [hit.doc_id for hit in hits if hit.system_id != source_id]
    if mismatched:
        raise ValueError(
            f"Candidate hits for source {source_id!r} contain mismatched documents: "
            + ", ".join(mismatched[:5])
        )


def _merge_metadata(primary: dict, secondary: dict) -> dict:
    merged = {**secondary, **primary}

    for field in ("matched_queries", "matched_query_indexes", "matched_keywords"):
        values: list = []
        seen: set = set()
        for source in (primary, secondary):
            raw_values = source.get(field)
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)
        if values:
            merged[field] = values

    primary_highlight = primary.get("highlight")
    secondary_highlight = secondary.get("highlight")
    if isinstance(primary_highlight, dict) or isinstance(secondary_highlight, dict):
        highlight: dict[str, list[str]] = {}
        for field in ("summary", "keywords"):
            values: list[str] = []
            seen_values: set[str] = set()
            for source in (primary_highlight, secondary_highlight):
                if not isinstance(source, dict):
                    continue
                raw_values = source.get(field)
                if not isinstance(raw_values, list):
                    continue
                for value in raw_values:
                    text = str(value)
                    if text in seen_values:
                        continue
                    seen_values.add(text)
                    values.append(text)
            if values:
                highlight[field] = values
        if highlight:
            merged["highlight"] = highlight

    return merged
