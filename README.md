# Generative AI & Agentic Workflows — Portfolio

Welcome — this repository is my living portfolio of experiments, demos, and reference implementations in generative AI and agentic orchestration. It highlights working prototypes, research-oriented notebooks, and utilities I use to prototype advanced LLM workflows.

> Status: Active work-in-progress. I update notebooks and add demos frequently; think of this as a lab notebook and portfolio combined.

---

## Quick highlights

- Focus on agentic and multi-agent architectures (LangChain, LangGraph)
- Practical demos for Retrieval-Augmented Generation (RAG) and vector-search integrations
- Notebooks demonstrating robust pipeline design: error handling, prompt patterns, and local model setups

---

## Portfolio structure (select highlights)

- LangGraphDemo/
  - MultiAgentArchitecture/ — reflection agents, communicating agents, Tree-of-Thought demos
  - Stateful and graph-based orchestrations showcasing conditional routing and iterative reasoning

- RAG/
  - End-to-end retrieval + LLM pipelines, document chunking, vector DB connectors, and re-ranking strategies

- Tools/
  - Custom tool wrappers and integrations used by agents (web ingestion, API wrappers, scrapers)

- Notebooks (sample)
  - Prompt_templates.ipynb — structured prompt engineering, Chain-of-Thought patterns, output parsers
  - ErrorHandling.ipynb — retry/backoff patterns, rate-limit handling, validation strategies
  - Local_models_demo.ipynb — running and integrating local/open models (Hugging Face / Ollama)

This repo mixes polished demos with exploratory notebooks; some files are prototype code and not production hardened — please review before reuse.

---

## Polished portfolio sections (what I can add next)
If you want this README to act like a formal portfolio landing page, I can add:
- A curated demo gallery (short descriptions + links to runnable notebooks)
- Status badges (CI, Python version, Docker image, coverage)
- A Roadmap / Milestones section showing planned features and upcoming notebooks
- Installation scripts and a requirements.txt / environment.yml for reproducible execution

Tell me which items above you'd like and I will add them.

---

## Getting started (recommended)
1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/ShivaUlavi/GenAI.git
cd GenAI
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

2. Install core tools (I add a requirements file on request):

```bash
pip install jupyterlab
# pip install -r requirements.txt  # when requirements file is present
jupyter lab
```

3. Environment variables
Some notebooks require external API keys or tokens. Configure these in your environment or a .env file (DO NOT commit secrets):

- AICREDITS_API_KEY, AICREDITS_BASE_URL — used by some examples
- HF_TOKEN — (optional) for Hugging Face model downloads
- TAVILY_API_KEY — (optional) for Tavily Search demonstrations

---

## Responsible use & security
- Never add API keys to commits; use environment variables or a secrets manager.
- Review notebooks before running — some demos call external services or web APIs.
- These examples are research/educational; not intended for production without further hardening.

---

## Contributing & contact
This is my personal portfolio-repo but contributions are welcome.
- Open an issue to request a feature or report a bug
- Send a PR with a short description of the change

Contact / profile: https://github.com/ShivaUlavi
Email: shivachetanulavi@gmail.com

---
