# AI Customer Support Copilot (RAG)

This project is a simple Retrieval-Augmented Generation (RAG) app for answering support questions from your own documents such as PDFs, FAQs, and product manuals.

## What It Does

1. Loads documents from `data/docs/`
2. Splits them into chunks
3. Converts chunks into embeddings
4. Stores embeddings in FAISS
5. Retrieves relevant chunks for a user query
6. Generates an answer using an OpenAI-compatible LLM endpoint

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI-compatible API
- FAISS
- PyPDF

## Project Structure

```text
genAI/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── docs/
│       └── .gitkeep
└── vectorstore/
    └── .gitkeep
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If your `pip` is old, upgrade it first:

```bash
python -m pip install --upgrade pip
```

Copy `.env.example` to `.env` and set your API key.

For NVIDIA-hosted models:

```env
NVIDIA_API_KEY=your_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=openai/gpt-oss-20b
NVIDIA_EMBEDDING_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
```

OpenAI also works as a fallback if you prefer.

## Add Your Documents

Place your support documents in `data/docs/`.

Supported formats:

- `.pdf`
- `.txt`
- `.md`

## Run The App

```bash
python -m streamlit run app.py
```

## How The Pipeline Works

1. Documents are loaded from disk
2. Text is split into overlapping chunks
3. Chunks are embedded with the configured embedding model
4. Embeddings are stored in FAISS
5. User question is matched against similar chunks
6. Retrieved context is passed to the chat model
7. The final answer is generated with source-backed context
