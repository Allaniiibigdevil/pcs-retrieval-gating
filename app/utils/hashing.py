import hashlib


def stable_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()
