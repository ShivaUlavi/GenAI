# Generative AI & Agentic Workflows Hub

An enterprise-ready repository focused on building, evaluating, and deploying production-grade Large Language Model (LLM) applications. This codebase demonstrates advanced orchestrations using LangChain and LangGraph, focusing on resilient, stateful, and autonomous AI agents.

## 🛠️ Repository Architecture

*   **`LangGraphDemo/`**: Implementation of advanced Agentic workflows. Showcases stateful multi-agent systems, conditional routing, and cyclic graphs where agents dynamically determine next steps based on tool outputs.
*   **`RAG/`**: Advanced Retrieval-Augmented Generation architectures. Covers document chunking strategies, vector database integrations, semantic search, and re-ranking pipelines to mitigate LLM hallucinations.
*   **`Tools/`**: Custom tool engineering enabling LLMs to securely interface with external APIs, web scrapers, and local system environments.
*   **`ErrorHandling.ipynb`**: Resiliency engineering for GenAI pipelines. Implements custom exception handlers, token rate-limit retry logic, API timeout fallbacks, and structured data validation.
*   **`Prompt_templates.ipynb`**: Structured prompt engineering frameworks focusing on Few-Shot prompting, Chain-of-Thought (CoT) reasoning, and output parsing to enforce JSON schema constraints.
*   **`Local_models_demo.ipynb`**: Prototyping LLM architectures locally using open-source models (Ollama/Hugging Face), optimized for restricted corporate network environments.

## 🧰 Core Tech Stack

*   **Frameworks:** LangChain, LangGraph
*   **Languages:** Python (Pandas, NumPy, BeautifulSoup, Requests), SQL, Bash
*   **Core Mechanics:** Vector Search, Stateful Memory Management, Automated Web Ingestion, Dynamic Routing

## 🚀 Key Architectural Focus Areas

1.  **State Management in Multi-Agent Systems:** Utilizing LangGraph to design complex graphs with clear nodes, conditional edges, and robust context retention across asynchronous agent interactions.
2.  **Pipeline Resiliency:** Building specialized error-handling checkpoints to ensure agentic loops do not enter infinite cycles or crash during live API failures.
3.  **Data Transformation:** Transforming unstructured text and web data (via custom scraping pipelines) into clean, vector-ready semantic tokens.
