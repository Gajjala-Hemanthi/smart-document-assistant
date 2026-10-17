# Smart Document Assistant

A document question-answering application built with Retrieval-Augmented Generation (RAG). Upload documents, ask questions in natural language, and receive answers based on retrieved document evidence.

## Features

- 📄 **Document Upload and Processing**: Supports document formats implemented in the application, including PDF, TXT, and spreadsheets.
- 🔍 **Text Retrieval**: Finds relevant document passages for a user's question.
- 🧠 **Vector Embeddings**: Uses Ollama's `nomic-embed-text` model to represent text as embeddings.
- 💾 **FAISS Vector Search**: Searches for semantically relevant document chunks.
- 🤖 **Local RAG Question Answering**: Uses the local `llama3.2:1b` model through Ollama to generate answers from retrieved context.
- 🎨 **Streamlit Interface**: Interact with the application through a web UI.
- 📚 **Source Evidence**: Inspect retrieved evidence used to answer questions.
- 📊 **Spreadsheet Support**: Includes spreadsheet-oriented handling implemented in the application.

## Technology Stack

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- `nomic-embed-text` — embedding model
- `llama3.2:1b` — local language model

## Prerequisites

- Python installed
- [Ollama](https://ollama.com/) installed and running
- Git (to clone the repository)
- Enough local resources to run the selected Ollama models

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gajjala-Hemanthi/smart-document-assistant.git
   cd smart-document-assistant
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the Ollama models used by the project:

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```

5. Start the application:

   ```bash
   streamlit run main.py
   ```

6. Open the local URL printed in the terminal by Streamlit.

## Usage

1. Start the application.
2. Upload a supported document through the interface.
3. Enter a question about the document.
4. Review the generated answer and its retrieved source evidence.

Answer quality depends on the information present in the uploaded documents and the relevance of the retrieved passages.

## Architecture

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Processing** — extract text from uploaded documents.
2. **Chunking** — divide extracted text into smaller passages.
3. **Embedding Generation** — convert passages into vector representations using `nomic-embed-text`.
4. **Vector Search** — use FAISS to retrieve passages relevant to the user's question.
5. **Context Assembly** — provide retrieved passages as context for answer generation.
6. **Answer Generation** — use `llama3.2:1b` through Ollama to generate a response grounded in the retrieved context.
7. **Evidence Display** — show source evidence in the interface so users can inspect the supporting text.

## Project Structure

```text
smart-document-assistant/
├── main.py
├── retrieval.py
├── requirements.txt
├── test_questions.py
├── README.md
└── .gitignore
```

## Testing

The repository includes `test_questions.py` for project-specific question tests. Run it with:

```bash
python test_questions.py
```

Make sure any test documents or configuration expected by the test script are available before running it.

## Security Notes

- Do not commit `.env` files, API keys, passwords, or other secrets.
- Review `.gitignore` before pushing changes to GitHub.
- Uploaded documents may contain private information; only use documents you are authorized to process.

## Future Improvements

- Add automated tests for retrieval quality and answer grounding.
- Improve handling of conflicting information across documents.
- Add clearer setup and troubleshooting guidance for different environments.

## Acknowledgments

This project uses open-source tools and libraries including Streamlit, LangChain, FAISS, and Ollama.


