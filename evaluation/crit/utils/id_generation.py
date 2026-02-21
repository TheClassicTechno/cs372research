import hashlib


def generate_reason_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
