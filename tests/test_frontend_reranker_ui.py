from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_displays_reranker_and_retrieval_evidence() -> None:
    app_source = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "Reranker" in app_source
    assert "BM25" in app_source
    assert "Vector" in app_source
    assert "matched_queries" in app_source
    assert "data.latency_ms?.total" in app_source
    assert "ES_TOP_K_DOCS" not in app_source
    assert "RERANKER_SCORE_THRESHOLD" not in app_source


def test_settings_page_points_to_actual_configuration_files() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "config/sources.json" in html
    assert "source.reranker_score_threshold" in html
    assert "RERANKER_MODEL_PATH" in html
    assert "RERANKER_BATCH_SIZE" in html
    assert "ES_TOP_K_DOCS" not in html
    assert "RERANKER_SCORE_THRESHOLD" not in html
