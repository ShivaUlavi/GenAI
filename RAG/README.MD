# Corporate Documentation Manager with Citations

An interactive, localized Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and LangGraph. This tool allows users to upload corporate, legal, or regulatory documents and query them in natural language, receiving precise answers backed by exact source citations.

## Features

* **Dynamic Document Ingestion:** Upload and process text-based corporate guides, policies, or regulatory documents directly through the browser interface.
* **Persistent Vector Store:** Utilizes Streamlit's `st.session_state` to maintain in-memory vector embeddings across UI interactions and script reruns.
* **LangGraph Orchestration:** Employs a stateful graph pipeline to handle retrieval and generation cleanly.
* **LLM Gateway Integration:** Powered by Llama 3.3 (70B) via an AICredits proxy with automated retry logic and timeout controls.

---

## Prerequisites & Environment Setup

This project uses custom API endpoints and keys managed via environment variables.

1. Obtain your API key from [AICredits.com](https://aicredits.com).
2. Set the required environment variables in your terminal before launching the application:

**For Command Prompt (CMD):**
```cmd
set AICREDITS_API_KEY=your_actual_api_key_here
set AICREDITS_BASE_URL=your_actual_base_url_here

**For Powershell:**
```cmd
$env:AICREDITS_API_KEY="your_actual_api_key_here"
$env:AICREDITS_BASE_URL="your_actual_base_url_here"

## Project Structure

```text
├── streamlit_app_V3.py    # Main Streamlit web application UI and event loop
├── rag.py                 # LangGraph workflow definition coordinating retrieval and generation
├── retriever.py           # Custom BaseRetriever handling document chunking and session persistence
├── document_loader.py     # Handles parsing and loading uploaded document files
└── llms.py                # Configures embedding stores and ChatOpenAI model connections

## How to Run

1. Ensure all five core files are saved in the exact same directory.
2. Open your Command Prompt (CMD) and navigate to your project directory.
3. Set the Python path to the current directory:
   ```cmd
   set PYTHONPATH=.
  streamlit run streamlit_app_V3.py
