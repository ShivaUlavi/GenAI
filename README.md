# Generative AI & Agentic Workflows Hub

Welcome — this repository is my central hub for experiments, demos, and research work in generative AI and agentic workflows. I'm actively developing and adding materials (notebooks, demos, and utilities) — expect the contents to change over time.

> Status: Work in progress — I'll be uploading new notebooks, tools, and examples as I continue building.

## 🔎 Overview

This repo collects hands-on examples and reference implementations around modern GenAI patterns and multi-agent orchestration.

### Primary focus
- Agentic architectures and orchestrations
- LangChain / LangGraph based workflows and stateful reasoning graphs
- Retrieval-Augmented Generation (RAG) patterns and vector search
- Prompt engineering, structured outputs, and robust pipeline error handling

## 📂 Repository layout
- `LangGraphDemo/` — advanced agentic workflows and multi-agent examples (state graphs, reflection agents, Tree-of-Thought demos)
- `RAG/` — retrieval-augmented generation experiments, chunking, vector DB integrations, and re-ranking
- `Tools/` — custom tool integrations (web scraping, API wrappers, system helpers)
- Notebooks such as `Prompt_templates.ipynb`, `ErrorHandling.ipynb`, and `Local_models_demo.ipynb`

(I'm still adding content; more folders and notebooks will appear over time.)

## ⚙️ Basic setup & running notebooks
1. Install Python 3.10+ (I use 3.11 in my environment).
2. Create a virtual environment and install dependencies. If a `requirements.txt` is added, run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Run Jupyter Lab / Notebook to open the example notebooks:
   ```bash
   pip install jupyterlab
   jupyter lab
   ```

Notes:
- Some demos rely on external API keys and local model endpoints. Set the following environment variables as needed in your environment or a .env file:
  - AICREDITS_API_KEY
  - AICREDITS_BASE_URL
  - HF_TOKEN (for Hugging Face model pulls)
  - TAVILY_API_KEY (if using Tavily Search in examples)

## 🚧 Security & responsible usage
- Be careful with API keys and secrets — never commit them to the repo. Use environment variables or a secrets manager.
- The notebooks are experiments and may call external services; review code before running on production data.

## 🤝 Contributing
This repo is personal and under active development. If you'd like to contribute or report issues:
- Open an issue to suggest improvements or point out bugs
- Submit a pull request with a short description of changes

If you want help converting notebooks into runnable examples / CI workflows or bundling a requirements file, tell me what environment you prefer and I can add it.

## 📬 Contact
GitHub: https://github.com/ShivaUlavi

---

I'll keep this README updated as I add more content. If you want specific sections (installation scripts, badge/status, examples list, or a license) added now, tell me which and I will update them.