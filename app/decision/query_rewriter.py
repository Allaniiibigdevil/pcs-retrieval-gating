async def rewrite_queries(task: str) -> list[str]:
    """Identity placeholder for the online query-rewrite service."""

    return [task]


def prepare_queries(task: str, rewritten_queries: list[str]) -> list[str]:
    """Keep the original task first, then append unique non-blank rewrites."""

    queries: list[str] = []
    seen: set[str] = set()
    for query in [task, *rewritten_queries]:
        cleaned = query.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        queries.append(cleaned)
    if not queries:
        raise ValueError("At least one non-blank query is required")
    return queries
