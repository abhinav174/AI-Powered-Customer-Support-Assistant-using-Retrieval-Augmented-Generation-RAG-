import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


class OpenAIEmbeddingClient(Embeddings):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _requires_asymmetric_mode(self) -> bool:
        normalized = self.model.lower()
        return "embedqa" in normalized or normalized.endswith("e5") or "-e5-" in normalized

    def _model_for_mode(self, mode: str) -> str:
        if not self._requires_asymmetric_mode():
            return self.model
        if self.model.endswith("-query") or self.model.endswith("-passage"):
            return self.model
        return f"{self.model}-{mode}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self._model_for_mode("passage"),
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self._model_for_mode("query"),
            input=[text],
        )
        return response.data[0].embedding


def get_provider_config() -> Dict[str, str]:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if nvidia_api_key:
        return {
            "provider": "nvidia",
            "api_key": nvidia_api_key,
            "base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "chat_model": os.getenv("NVIDIA_CHAT_MODEL", "openai/gpt-oss-20b"),
            "embedding_model": os.getenv(
                "NVIDIA_EMBEDDING_MODEL",
                "nvidia/nv-embed-v1",
            ),
        }

    if openai_api_key:
        return {
            "provider": "openai",
            "api_key": openai_api_key,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        }

    raise ValueError(
        "No API key found. Add NVIDIA_API_KEY or OPENAI_API_KEY to your .env file."
    )


def get_supported_files(directory: Path) -> List[Path]:
    supported_extensions = {".pdf", ".txt", ".md"}
    return [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]


def load_documents(directory: Path = DOCS_DIR) -> List[Document]:
    documents: List[Document] = []

    for file_path in get_supported_files(directory):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = str(file_path.name)
        documents.extend(loaded_docs)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> Embeddings:
    config = get_provider_config()

    return OpenAIEmbeddingClient(
        model=config["embedding_model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


def build_vectorstore() -> Tuple[FAISS, int]:
    documents = load_documents()
    if not documents:
        raise ValueError("No supported documents found in data/docs.")

    chunks = split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore, len(chunks)


def load_or_create_vectorstore() -> Tuple[FAISS, bool, int]:
    index_file = VECTORSTORE_DIR / "index.faiss"
    embeddings = get_embeddings()

    if index_file.exists():
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, False, 0

    vectorstore, chunk_count = build_vectorstore()
    return vectorstore, True, chunk_count


def rebuild_vectorstore() -> int:
    _, chunk_count = build_vectorstore()
    return chunk_count


def generate_answer(question: str, top_k: int = 4) -> dict:
    vectorstore, _, _ = load_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
            "context": [],
        }

    context_blocks = []
    source_names = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        context_blocks.append(f"Source: {source}\nContent: {doc.page_content}")
        source_names.append(source)

    prompt = f"""
You are an AI customer support copilot.
Answer the user's question using only the provided context.
If the answer is not in the context, clearly say you could not find it in the documents.
Keep the answer helpful, concise, and accurate.

Context:
{chr(10).join(context_blocks)}

Question:
{question}
""".strip()

    config = get_provider_config()
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    response = client.chat.completions.create(
        model=config["chat_model"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI customer support copilot. "
                    "Answer using only the provided document context. "
                    "If the answer is missing, say you could not find it in the documents."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sorted(set(source_names)),
        "context": retrieved_docs,
    }
