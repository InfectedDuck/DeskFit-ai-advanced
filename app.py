"""DeskFit AI - Streamlit UI for the RAG-powered micro-fitness assistant."""

import streamlit as st

import config
from src.embeddings import EmbeddingsClient
from src.hybrid_retriever import HybridRetriever
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.vector_db import VectorDatabase

# --- Page Configuration ---
st.set_page_config(
    page_title="DeskFit AI",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Initialize Components (cached) ---
@st.cache_resource
def init_embeddings() -> EmbeddingsClient:
    return EmbeddingsClient(config.EMBEDDING_MODEL_NAME)


@st.cache_resource
def init_vector_db() -> VectorDatabase:
    return VectorDatabase(config.CHROMA_DB_DIR, config.COLLECTION_NAME)


@st.cache_resource
def init_llm_client() -> LLMClient:
    return LLMClient(config.DIAL_API_BASE, config.DIAL_API_KEY, config.DIAL_MODEL_NAME)


@st.cache_resource
def init_hybrid_retriever(
    _embeddings: EmbeddingsClient, _db: VectorDatabase
) -> HybridRetriever:
    return HybridRetriever(
        embeddings_client=_embeddings,
        vector_db=_db,
        vector_k=config.VECTOR_CANDIDATE_K,
        bm25_k=config.BM25_CANDIDATE_K,
        final_k=config.TOP_K_RESULTS,
    )


def init_rag_pipeline(
    embeddings: EmbeddingsClient,
    db: VectorDatabase,
    llm: LLMClient,
    hybrid_retriever: HybridRetriever | None = None,
) -> RAGPipeline:
    return RAGPipeline(embeddings, db, llm, hybrid_retriever=hybrid_retriever)


# --- Sidebar ---
def render_sidebar() -> dict | None:
    """Render the sidebar with status, filters, and example questions."""
    with st.sidebar:
        st.title("DeskFit AI")
        st.caption("Micro-Fitness for the Workaholic")

        # System status
        st.subheader("System Status")

        db = init_vector_db()
        doc_count = db.count()

        if doc_count > 0:
            st.success(f"Vector DB: {doc_count} documents")
        else:
            st.error("Vector DB: Empty — run `python -m src.populate_db`")

        if config.DIAL_API_KEY:
            st.success(f"LLM: {config.DIAL_MODEL_NAME}")
        else:
            st.warning("LLM: No API key set in .env")

        st.success(f"Embeddings: {config.EMBEDDING_MODEL_NAME}")

        st.divider()

        # Filters
        st.subheader("Search Filters")

        content_type = st.selectbox(
            "Content type",
            ["All", "Exercise", "Posture Tip", "Wellness Advice"],
        )

        body_area = st.selectbox(
            "Body area (exercises only)",
            ["All", "neck", "shoulders", "back", "wrists", "eyes", "legs", "full_body"],
        )

        difficulty = st.selectbox(
            "Difficulty (exercises only)",
            ["All", "beginner", "intermediate"],
        )

        # Build filter dict
        filters = None
        filter_conditions = []

        if content_type != "All":
            type_map = {
                "Exercise": "exercise",
                "Posture Tip": "posture_tip",
                "Wellness Advice": "wellness_advice",
            }
            filter_conditions.append({"type": type_map[content_type]})

        if body_area != "All":
            filter_conditions.append({"body_area": body_area})

        if difficulty != "All":
            filter_conditions.append({"difficulty": difficulty})

        if len(filter_conditions) == 1:
            filters = filter_conditions[0]
        elif len(filter_conditions) > 1:
            filters = {"$and": filter_conditions}

        st.divider()

        # Example questions
        st.subheader("Try asking...")
        examples = [
            "My neck hurts from looking at screens all day",
            "Quick stretch I can do between meetings",
            "How should I set up my monitor?",
            "Breathing exercise for stress before a presentation",
            "My wrists hurt from typing too much",
            "I feel sleepy every afternoon at work",
            "Best exercises for lower back pain from sitting",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{example[:20]}", use_container_width=True):
                st.session_state["pending_question"] = example

        st.divider()

        # Show context toggle
        st.session_state["show_sources"] = st.toggle(
            "Show retrieved sources", value=st.session_state.get("show_sources", True)
        )

    return filters


# --- Main Chat Interface ---
def main() -> None:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None

    filters = render_sidebar()

    # Header
    st.title("DeskFit AI")
    st.caption(
        "Your AI wellness assistant for desk exercises, posture tips, and recovery strategies. "
        "Powered by RAG with a curated micro-fitness knowledge base."
    )

    # Initialize components
    embeddings = init_embeddings()
    db = init_vector_db()
    llm = init_llm_client()

    hybrid = None
    if config.USE_HYBRID_RETRIEVAL:
        hybrid = init_hybrid_retriever(embeddings, db)

    pipeline = init_rag_pipeline(embeddings, db, llm, hybrid_retriever=hybrid)

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                if st.session_state.get("show_sources", True):
                    render_sources(msg["sources"])

    # Handle input (from chat box or example button)
    user_input = st.chat_input("Ask about desk exercises, posture, or wellness...")

    if st.session_state.get("pending_question"):
        user_input = st.session_state.pop("pending_question")

    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate RAG response
        with st.chat_message("assistant"):
            if db.count() == 0:
                st.warning(
                    "The knowledge base is empty. Please run "
                    "`python -m src.populate_db` first."
                )
                return

            stream, sources = pipeline.query_stream(
                user_input, top_k=config.TOP_K_RESULTS, filters=filters
            )

            response_text = st.write_stream(stream)

            if st.session_state.get("show_sources", True):
                render_sources(sources)

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text,
            "sources": sources,
        })


def render_sources(sources: list[dict]) -> None:
    """Render retrieved source chunks in an expander."""
    if not sources:
        return

    with st.expander(f"Retrieved Sources ({len(sources)} chunks)", expanded=False):
        for i, chunk in enumerate(sources, 1):
            metadata = chunk.get("metadata", {})
            distance = chunk.get("distance")
            rerank_score = chunk.get("rerank_score")
            if rerank_score is not None:
                similarity = max(0, min(1, (rerank_score + 10) / 20))  # normalize approx
            elif distance is not None:
                similarity = max(0, 1 - distance)
            else:
                similarity = 0

            doc_type = metadata.get("type", "unknown").replace("_", " ").title()
            title = metadata.get("title", "Untitled")

            st.markdown(f"**[{i}] {doc_type}: {title}** — Relevance: {similarity:.0%}")

            # Show key metadata
            meta_items = []
            for key in ["category", "body_area", "difficulty", "duration_seconds", "applies_to", "context"]:
                if key in metadata:
                    meta_items.append(f"`{key}: {metadata[key]}`")
            if meta_items:
                st.markdown(" | ".join(meta_items))

            st.divider()


if __name__ == "__main__":
    main()
