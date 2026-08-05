from app.decision.query_rewriter import prepare_queries


def test_prepare_queries_keeps_original_task_first_and_deduplicates() -> None:
    queries = prepare_queries(
        "  original task  ",
        ["rewrite one", "original task", "rewrite one", "  ", "rewrite two"],
    )

    assert queries == ["original task", "rewrite one", "rewrite two"]
