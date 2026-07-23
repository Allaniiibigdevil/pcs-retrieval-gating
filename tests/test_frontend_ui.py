from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_shows_rewritten_queries_newest_first() -> None:
    app_source = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "data.rewritten_queries" in app_source
    assert "messageList.prepend(node)" in app_source
    assert "messageList.lastElementChild" not in app_source


def test_settings_page_documents_evidence_limit_and_multi_query_retrieval() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "EVIDENCE_DOCS_PER_SYSTEM" in html
    assert "每条改写 query 只执行一次 FAISS search" in html
    assert "ES_TOP_K_DOCS" in html
    assert "FAISS_TOP_K_DOCS" in html
    assert "RRF_ES_ONLY_WEIGHT" in html
    assert "w_es(doc)" in html


def test_frontend_displays_document_and_system_rrf_scores() -> None:
    app_source = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "RRF 分数" in app_source
    assert "最佳 RRF 分数" in app_source
    assert "RRF_ES_ONLY_WEIGHT" in app_source
