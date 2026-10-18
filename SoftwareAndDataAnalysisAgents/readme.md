## 🤖 RAG Code Generation & Validation Agent

This module implements an advanced agentic RAG and self-correction pipeline designed to combat LLM code hallucinations through a robust **Generator-Critic (Code-Auditor)** architecture.

### 🌟 Key Capabilities
* **Codebase RAG:** Automatically clones and indexes repository notebooks via `GitLoader`, semantic code splitting, and Maximal Marginal Relevance (MMR) vector search (`Chroma`).
* **Context-Aware Generation:** Synthesizes custom workflow implementations using Llama-3.1-70B grounded in your repository's patterns.
* **Automated Validation Loop:** Passes generated code directly to an independent critic (Claude) to aggressively audit for framework hallucinations, bad API signatures, and logic flaws—delivering immediate code critique and production-ready corrections.
