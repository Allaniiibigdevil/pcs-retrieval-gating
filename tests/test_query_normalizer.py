from app.decision.query_normalizer import QueryNormalizer


def test_query_normalizer_preserves_stop_words_for_es_analyzer() -> None:
    normalizer = QueryNormalizer()

    assert normalizer.normalize("  帮我 查一下 我的 海鲜过敏  ") == "帮我 查一下 我的 海鲜过敏"


def test_query_normalizer_only_collapses_whitespace() -> None:
    normalizer = QueryNormalizer()

    assert normalizer.normalize("请\n查询\tAlbum") == "请 查询 Album"
