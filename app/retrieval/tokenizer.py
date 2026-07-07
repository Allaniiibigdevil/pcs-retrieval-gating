import re
import warnings

from app.retrieval.lexicon import load_stopwords, load_synonyms
from app.schemas.doc import SourceDoc


def tokenize(text: str) -> list[str]:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module="jieba\\._compat",
    )
    import jieba

    jieba.setLogLevel(30)
    normalized = text.strip().lower()
    tokens: list[str] = []
    seen: set[str] = set()
    stopwords = load_stopwords()
    synonyms = load_synonyms()
    for token in jieba.lcut(normalized, cut_all=False):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"\W+", token):
            continue
        if token in stopwords:
            continue
        _append_token(tokens, seen, token)
        for synonym in synonyms.get(token, []):
            if synonym not in stopwords:
                _append_token(tokens, seen, synonym)

    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        for token in _char_ngrams(run):
            if token in stopwords:
                continue
            _append_token(tokens, seen, token)
            for synonym in synonyms.get(token, []):
                if synonym not in stopwords:
                    _append_token(tokens, seen, synonym)
    return tokens


def _append_token(tokens: list[str], seen: set[str], token: str) -> None:
    if token not in seen:
        seen.add(token)
        tokens.append(token)


def _char_ngrams(text: str) -> list[str]:
    if len(text) <= 1:
        return [text]

    grams = [text]
    for size in (2, 3):
        if len(text) >= size:
            grams.extend(text[index : index + size] for index in range(len(text) - size + 1))
    return grams


def build_bm25_text(doc: SourceDoc) -> str:
    weighted_keywords = " ".join(doc.keywords * 3)
    return f"{doc.system_id} {doc.summary} {weighted_keywords}"
