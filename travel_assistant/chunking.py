def chunk_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 1,
):

    text = text.strip()

    if not text:
        return []

    sentences = text.split(". ")

    cleaned_sentences = []

    for s in sentences:

        s = s.strip()

        if not s:
            continue

        if not s.endswith("."):
            s += "."

        cleaned_sentences.append(s)

    chunks = []

    current_chunk = []

    current_length = 0

    for sentence in cleaned_sentences:

        sentence_length = len(sentence)

        if (
            current_chunk
            and current_length + sentence_length > chunk_size
        ):

            chunks.append(" ".join(current_chunk))

            # overlap
            current_chunk = current_chunk[-overlap:]

            current_length = sum(
                len(s)
                for s in current_chunk
            )

        current_chunk.append(sentence)

        current_length += sentence_length

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks