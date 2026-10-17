\# Smart Document Assistant



A RAG-based application that answers questions using information retrieved from uploaded documents.



\## Features



\- Upload and process supported documents.

\- Ask questions in natural language.

\- Retrieve relevant document passages using FAISS.

\- Generate answers using a local LLM through Ollama.

\- Display source evidence to help users verify answers.



\## Tech Stack



\- \*\*Python\*\* — application logic

\- \*\*Streamlit\*\* — user interface

\- \*\*LangChain\*\* — document processing and integrations

\- \*\*FAISS\*\* — vector similarity search

\- \*\*Ollama\*\* — local language model and embeddings



\## How RAG Works



1\. Extract text from documents.

2\. Split text into smaller chunks.

3\. Convert chunks into embeddings.

4\. Store and search embeddings using FAISS.

5\. Retrieve relevant chunks for a user's question.

6\. Send the retrieved context to the LLM to generate an answer.



\## Project Structure



```text

smart\_document\_assistant/

├── main.py

├── retrieval.py

├── requirements.txt

├── test\_questions.py

├── README.md

└── .gitignore

```



\## Setup and Run



1\. Install Python and \[Ollama](https://ollama.com/).

2\. Download the required Ollama models:



&#x20;  ```bash

&#x20;  ollama pull nomic-embed-text

&#x20;  ollama pull llama3.2:1b

&#x20;  ```



3\. Install the Python dependencies:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Start the application:



&#x20;  ```bash

&#x20;  streamlit run main.py

&#x20;  ```



5\. Open the local URL shown in your terminal and upload a document to begin asking questions.



\## Testing



Run the project's test script, if its required test inputs and configuration are available:



```bash

python test\_questions.py

```



\## Purpose



This project demonstrates how Retrieval-Augmented Generation (RAG) can connect document retrieval with a local language model to answer questions using document evidence.

