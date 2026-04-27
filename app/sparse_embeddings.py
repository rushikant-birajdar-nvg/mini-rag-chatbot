import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_sparse_vector(text: str, max_terms: int = 64) -> dict[str, list[int] | list[float]]:
    tokens = _tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    most_common = counts.most_common(max_terms)
    max_count = most_common[0][1]

    indices: list[int] = []
    values: list[float] = []
    for token, count in most_common:
        index = abs(hash(token)) % 1_000_003
        indices.append(index)
        values.append(count / max_count)

    return {"indices": indices, "values": values}


def make_sparse_vectors(texts: list[str], max_terms: int = 64) -> list[dict[str, list[int] | list[float]]]:
    return [make_sparse_vector(text, max_terms=max_terms) for text in texts]
