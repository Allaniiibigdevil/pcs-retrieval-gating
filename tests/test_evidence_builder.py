from app.decision.evidence_builder import EvidenceBuilder
from app.schemas.search import SearchHit


def test_evidence_builder_matches_keyword_by_bigram_fallback() -> None:
    hit = SearchHit(
        doc_id="3",
        system_id="notepad",
        summary="\u8bb0\u5f55\u4e86\u7528\u6237\u5bf9\u6d77\u9c9c\u8fc7\u654f",
        keywords=["\u8fc7\u654f", "\u6d77\u9c9c\u8fc7\u654f"],
    )

    enriched = EvidenceBuilder().build("\u6211\u53ef\u4ee5\u5403\u6d77\u9c9c\u5417\uff1f", [hit])

    assert enriched[0].metadata["matched_keywords"] == ["\u6d77\u9c9c\u8fc7\u654f"]


def test_evidence_builder_preserves_es_highlight_matched_keywords() -> None:
    hit = SearchHit(
        doc_id="3",
        system_id="notepad",
        summary="记录了用户对海鲜过敏",
        keywords=["过敏", "海鲜过敏"],
        metadata={"matched_keywords": ["过敏"]},
    )

    enriched = EvidenceBuilder().build("我可以吃海鲜吗？", [hit])

    assert enriched[0].metadata["matched_keywords"] == ["过敏"]
