from typing import List


def split_by_tokens(text: str, max_tokens: int = 300) -> List[str]:
# super simple splitter by chars ~ proxy for tokens
    chunk_size = max_tokens * 4
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + chunk_size, len(text))
        # try to break on paragraph
        k = text.rfind("\n\n", i, j)
        if k == -1:
            k = j
        chunks.append(text[i:k].strip())
        i = k
        return [c for c in chunks if c]