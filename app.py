import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import DOCS_DIR, generate_answer, rebuild_vectorstore


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(
    page_title="AI Customer Support Copilot",
    page_icon="💬",
    layout="wide",
)


def ensure_data_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    Path("vectorstore").mkdir(parents=True, exist_ok=True)


def get_env_debug_info() -> dict:
    return {
        "env_file_exists": ENV_PATH.exists(),
        "nvidia_api_key_loaded": bool(os.getenv("NVIDIA_API_KEY")),
        "openai_api_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "nvidia_base_url": os.getenv("NVIDIA_BASE_URL", ""),
        "nvidia_chat_model": os.getenv("NVIDIA_CHAT_MODEL", ""),
        "nvidia_embedding_model": os.getenv("NVIDIA_EMBEDDING_MODEL", ""),
    }


ensure_data_dirs()

st.title("AI Customer Support Copilot (RAG)")
st.caption("Ask support questions grounded in your own PDFs, FAQs, and product docs.")

with st.sidebar:
    st.subheader("Knowledge Base")
    st.write(f"Document folder: `{DOCS_DIR}`")

    files = sorted(
        [
            path.name
            for path in DOCS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
    )

    if files:
        st.write("Loaded files:")
        for file_name in files:
            st.write(f"- {file_name}")
    else:
        st.warning("No documents found. Add PDFs, TXT, or MD files to `data/docs/`.")

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding vector database..."):
            try:
                chunk_count = rebuild_vectorstore()
                st.success(f"Knowledge base rebuilt with {chunk_count} chunks.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Environment Debug")
    env_debug = get_env_debug_info()
    st.write(f".env exists: `{env_debug['env_file_exists']}`")
    st.write(f"NVIDIA key loaded: `{env_debug['nvidia_api_key_loaded']}`")
    st.write(f"OpenAI key loaded: `{env_debug['openai_api_key_loaded']}`")
    if env_debug["nvidia_base_url"]:
        st.write(f"NVIDIA base URL: `{env_debug['nvidia_base_url']}`")
    if env_debug["nvidia_chat_model"]:
        st.write(f"NVIDIA chat model: `{env_debug['nvidia_chat_model']}`")
    if env_debug["nvidia_embedding_model"]:
        st.write(f"NVIDIA embedding model: `{env_debug['nvidia_embedding_model']}`")

st.markdown(
    """
### How it works
1. Your documents are split into chunks
2. Chunks are embedded and stored in FAISS
3. Relevant chunks are retrieved for each question
4. The LLM answers using only the retrieved context
"""
)

question = st.chat_input("Ask a question about your company docs...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = generate_answer(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                if sources:
                    st.markdown("**Sources:** " + ", ".join(sources))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer + (
                            f"\n\n**Sources:** {', '.join(sources)}" if sources else ""
                        ),
                    }
                )
            except Exception as exc:
                error_message = str(exc)
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {error_message}"}
                )
