async def rewrite_queries(task: str) -> list[str]:
    """Return one or more retrieval queries for the original task."""

    # TODO: Replace this identity implementation with query rewriting logic.
    return [task]


def prepare_queries(task: str, rewritten_queries: list[str]) -> list[str]:
    """Keep the original task first, then append non-blank unique rewrites."""

    queries: list[str] = []
    seen: set[str] = set()
    for query in [task, *rewritten_queries]:
        cleaned = query.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        queries.append(cleaned)
    return queries or [task]
