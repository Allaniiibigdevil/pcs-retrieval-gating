import re


class QueryNormalizer:
    def normalize(self, task: str) -> str:
        """Keep query text intact while removing only accidental extra whitespace.

        Tokenization, lowercasing, stop words, synonyms, and domain vocabulary belong to
        the configured Elasticsearch analyzer so ES scores stay reproducible.
        """
        return re.sub(r"\s+", " ", task.strip())
