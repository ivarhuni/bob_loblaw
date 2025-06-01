# Icelandic Law Q&A System

A local CLI application for asking questions about Icelandic law using RAG (Retrieval Augmented Generation).

## Features

- 🔍 Vector search for relevant legal documents
- 🧠 Local LLM for answering questions
- 📚 Context-aware responses with source citations
- 💻 Fully local operation - no external APIs needed

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the Mistral 7B model:
   - Create a `models` directory
   - Download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` from HuggingFace
   - Place it in the `models` directory

3. Build the vector store:
   ```bash
   python build_law_vectors.py
   ```

## Usage

Run the Q&A system:
```bash
python ask_law.py
```

Type your questions in Icelandic and get answers with relevant legal citations.

Example questions:
- "Hverjar eru helstu skyldur atvinnurekanda gagnvart starfsmönnum?"
- "Hvaða réttindi hafa foreldrar gagnvart börnum sínum?"
- "Hvernig er með réttindi fatlaðra manna á vinnumarkaði?"

## How it Works

1. The system uses FAISS vector store to find relevant legal documents
2. The Mistral 7B model generates answers using the retrieved context
3. Responses include citations to the source laws and articles

## Requirements

- Python 3.8+
- 16GB+ RAM recommended
- 8GB+ free disk space 