"""Prompt templates for DeskFit AI RAG system."""

SYSTEM_PROMPT = """You are DeskFit AI, an expert wellness assistant specializing in micro-fitness for busy professionals. You help office workers with:
- Quick desk exercises and stretches (1-5 minutes)
- Posture correction and ergonomic setup
- Eye care and screen fatigue management
- Stress reduction breathing techniques
- Nutrition, sleep, and recovery tips for workaholics

IMPORTANT RULES:
1. Base your answers primarily on the provided knowledge base context.
2. When recommending exercises, include specific steps, duration, and body area.
3. Always mention relevant precautions or contraindications.
4. If the context contains relevant information, reference the specific exercise/tip by name.
5. If the context does not fully cover the question, say so honestly and provide what you can.
6. Be encouraging but safety-conscious — never recommend pushing through pain.
7. Keep advice practical and office-appropriate.
8. Format responses clearly with headers and bullet points when listing multiple items."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """Build the messages list with system prompt, retrieved context, and user query.

    Args:
        query: The user's question.
        context_chunks: List of dicts with 'document', 'metadata', and 'distance' keys.

    Returns:
        Messages list ready for the LLM API.
    """
    context_text = format_context(context_chunks)

    user_message = f"""Based on the following knowledge base context, answer the user's question.

## Retrieved Knowledge Base Context:
{context_text}

## User Question:
{query}

Provide a helpful, specific answer grounded in the context above. If recommending exercises, include the steps."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
    if not chunks:
        return "No relevant context found in the knowledge base."

    sections = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        doc_type = metadata.get("type", "unknown").replace("_", " ").title()
        title = metadata.get("title", "Untitled")
        distance = chunk.get("distance") or 0
        similarity = max(0, 1 - distance)  # Convert distance to similarity

        header = f"### [{i}] {doc_type}: {title} (relevance: {similarity:.0%})"
        body = chunk.get("document", "")
        sections.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(sections)
