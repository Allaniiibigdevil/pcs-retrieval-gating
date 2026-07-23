from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_uses_only_reranker_for_final_score_and_rank() -> None:
    app_source = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "最终 Reranker 分数" in app_source
    assert "最终 Reranker 排名" in app_source
    assert "粗召回信号（不参与最终排名）" in app_source
    assert "data.rewritten_queries" in app_source
    assert "messageList.prepend(node)" in app_source
    assert "rrf" not in app_source.lower()


def test_settings_page_documents_reranker_formula() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "sᵢ = sigmoid(zᵢ)" in html
    assert "reranker_rank(dᵢ) = rank_desc(sᵢ)" in html
    assert "Score(S) = max" in html
    assert "RERANKER_SCORE_THRESHOLD" in html
    assert "ES_TOP_K_DOCS" in html
    assert "FAISS_TOP_K_DOCS" in html
    assert "EVIDENCE_DOCS_PER_SYSTEM" in html
    assert "RERANKER_MAX_CANDIDATES" not in html
    assert "DEFAULT_TOP_K_DOCS" not in html
    assert "rrf" not in html.lower()
