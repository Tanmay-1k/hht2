SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer the user's question using ONLY the provided context passages.
Rules:
- If the context does not contain enough information to answer, say
  exactly: "I don't have enough information in the provided context to answer that."
- Do not use outside knowledge, even if you know the answer.
- Keep answers concise (2-4 sentences).
- After the answer, list the chunk_ids you actually used as evidence.
"""

ANSWER_TEMPLATE = """Context passages:
{context_block}

Question: {question}

Respond in this exact format:
ANSWER: <your answer>
CITED_CHUNK_IDS: <comma-separated chunk_ids you used, or NONE>
"""


def build_context_block(chunks) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] {c.text}")
    return "\n\n".join(lines)
