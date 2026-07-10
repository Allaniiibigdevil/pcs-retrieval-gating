from app.retrieval.tokenizer import build_keyword_text, tokenize
from app.schemas.doc import SourceDoc


def test_tokenize_keeps_chinese_keywords() -> None:
    tokens = tokenize("\u4e0a\u6d77\u51fa\u5dee\u4f1a\u8bae\u7167\u7247")

    assert "\u4e0a\u6d77" in tokens
    assert "\u51fa\u5dee" in tokens
    assert "\u4f1a\u8bae" in tokens


def test_tokenize_adds_chinese_ngram_fallback() -> None:
    tokens = tokenize("\u6211\u53ef\u4ee5\u5403\u6d77\u9c9c\u5417")

    assert "\u6d77\u9c9c" in tokens


def test_build_keyword_text_weights_keywords() -> None:
    doc = SourceDoc(
        doc_id="memo_doc_001",
        system_id="memo_system",
        summary="\u7528\u6237\u8bb0\u5f55\u4e86\u4e0a\u6d77\u51fa\u5dee\u8ba1\u5212\u3002",
        keywords=["\u4e0a\u6d77", "\u4f1a\u8bae"],
    )

    text = build_keyword_text(doc)

    assert "memo_system" in text
    assert "\u7528\u6237\u8bb0\u5f55\u4e86\u4e0a\u6d77\u51fa\u5dee\u8ba1\u5212\u3002" in text
    assert text.count("\u4e0a\u6d77") >= 3


def test_tokenize_applies_stopwords_and_synonyms(monkeypatch, tmp_path) -> None:
    stopwords_path = tmp_path / "stopwords.txt"
    stopwords_path.write_text("可以\n吗\n", encoding="utf-8")
    synonyms_path = tmp_path / "synonyms.txt"
    synonyms_path.write_text("海鲜,水产\n", encoding="utf-8")

    from app.config import get_settings
    from app.retrieval.lexicon import load_stopwords, load_synonyms

    settings = get_settings()
    monkeypatch.setattr(settings, "LOCAL_STOPWORDS_PATH", str(stopwords_path))
    monkeypatch.setattr(settings, "LOCAL_SYNONYMS_PATH", str(synonyms_path))
    load_stopwords.cache_clear()
    load_synonyms.cache_clear()

    tokens = tokenize("我可以吃海鲜吗")

    assert "可以" not in tokens
    assert "吗" not in tokens
    assert "海鲜" in tokens
    assert "水产" in tokens

    load_stopwords.cache_clear()
    load_synonyms.cache_clear()
