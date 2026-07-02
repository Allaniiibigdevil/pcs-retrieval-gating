import re


class QueryNormalizer:
    STOP_WORDS = [
        "我",
        "想",
        "查询",
        "查一下",
        "帮我",
        "请",
        "的",
        "了",
        "是",
        "一个",
        "这个",
        "那个",
    ]

    def normalize(self, task: str) -> str:
        normalized = task.strip().lower()
        for stop_word in sorted(self.STOP_WORDS, key=len, reverse=True):
            normalized = normalized.replace(stop_word, " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
