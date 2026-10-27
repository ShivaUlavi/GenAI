# GenAI Portfolio

I build and explore practical generative AI systems that combine language models, retrieval, tools, orchestration, and multimodal reasoning. This repository is my portfolio of experiments, prototypes, and applied demos in agentic AI and LLM workflows.

The work here focuses on turning ideas into working systems: from prompt engineering and retrieval pipelines to LangGraph agents, multimodal analysis, and tool-using AI applications.

> Status: Active research and prototyping workspace

## What I am building

This repo brings together several core areas of modern AI engineering:

- Agentic workflows with LangGraph and LangChain
- Retrieval-augmented generation (RAG) and grounded Q&A
- Tool calling, planning, and structured output generation
- Multimodal AI for images, video, and visual reasoning
- Local model experimentation and provider testing
- Validation loops for improving LLM reliability and output quality

## Portfolio themes

### 1. Agentic systems
I explore how LLMs can move beyond simple chat to orchestrated, stateful workflows with memory, branching logic, and tool use.

Highlights:
- LangGraph demos and graph-based orchestration
- Multi-agent architecture experiments
- Conversational and stateful agent patterns
- Output parsing and structured response design

### 2. Retrieval and knowledge systems
I build retrieval workflows that connect models to external knowledge and domain data.

Highlights:
- Document ingestion and chunking pipelines
- Retriever implementations for semantic search
- LLM + vector retrieval patterns
- Corporate documentation and grounded Q&A systems

### 3. Tools and workflow automation
These projects investigate how agents can act through tools, coordinate tasks, and produce reliable execution flows.

Highlights:
- Built-in and custom tool demos
- Structured outputs and tool node patterns
- Planning-first agent patterns
- Tool-enabled reasoning workflows

### 4. Multimodal AI
I also explore multimodal intelligence for visual tasks, exercise analysis, and video understanding.

Highlights:
- Sequential multimodal narrative generation
- Asynchronous multimodal evaluation workflows
- Workout and form analysis using vision models
- Local video understanding with Ollama and multimodal pipelines

### 5. Practical AI engineering
A large part of the repo is devoted to reliability, experimentation, and improving real-world usability of AI systems.

Highlights:
- Prompt pattern testing
- Retry/error handling and robustness patterns
- Local model inference checks
- LLM API token and setup validation
- Code-generation workflows with validation loops

## Repository structure

- `LangGraphDemo/` — LangGraph experiments, graph orchestration, and multi-agent demos
- `RAG/` — retrieval pipelines, retrievers, LLM integration, and app prototypes
- `Tools/` — tool use, planning, structured outputs, and agent tooling
- `MultiModalAgents/` — multimodal reasoning, vision workflows, and video analysis
- `SoftwareAndDataAnalysisAgents/` — code generation and analytical agent experiments
- `TestScripts/` — local model validation and provider/API tests
- `Prompt_templates.ipynb` — prompt-engineering experiments
- `ErrorHandling.ipynb` — reliability and failure-handling patterns
- `requirements.txt` — project dependencies

## Featured work

### RAG and knowledge retrieval
- `RAG/streamlit_app_V3.py` — interactive app for grounded document-based Q&A
- `RAG/rag.py` — main retrieval + generation orchestration flow
- `RAG/retriever.py` — custom retrieval logic and chunking approach

### LangGraph and agent orchestration
- `LangGraphDemo/SimpleLangGraphDemo.ipynb` — foundational graph workflow example
- `LangGraphDemo/GraphWithOutputParser.ipynb` — structured parsing within a graph flow
- `LangGraphDemo/MultiAgentArchitecture/` — collaborative agent architecture patterns

### Tools and agent capabilities
- `Tools/CustomToolsDemo.ipynb` — custom tool orchestration examples
- `Tools/PlanAndSolve.ipynb` — planning-driven AI workflows
- `Tools/StructuredOutput.ipynb` — schema-based generation patterns

### Multimodal AI
- `MultiModalAgents/01_sequential_multimodal_narrative_generator.ipynb` — narrative generation from visual input
- `MultiModalAgents/02_asynchronous_multimodal_portfolio_evaluator.ipynb` — multimodal evaluation workflow
- `MultiModalAgents/03_Local_Edge_Gym_Form_Analyzer_Ollama.ipynb` — local multimodal form analysis
- `MultiModalAgents/WorkoutAnalyzerApp/` — video upload and technique feedback app

### Software and analysis agents
- `SoftwareAndDataAnalysisAgents/RAG_Code_Generation_with_Validation_Loop.ipynb` — code generation with feedback-based validation
- `SoftwareAndDataAnalysisAgents/python_code_generation.ipynb` — generation patterns for Python tasks

## Tech stack

- Python
- LangChain
- LangGraph
- Streamlit
- Jupyter Notebooks
- Ollama
- OpenAI / Anthropic / Gemini / Hugging Face integrations
- Vector-store and retrieval workflows

## Getting started

1. Clone the repository

```bash
git clone https://github.com/ShivaUlavi/GenAI.git
cd GenAI
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Launch notebooks or run project-specific apps

```bash
jupyter lab
```

## Environment notes

Some examples require external API keys or local services. Common variables include:

- `AICREDITS_API_KEY`
- `AICREDITS_BASE_URL`
- `HF_TOKEN`
- `TAVILY_API_KEY`

Keep secrets in local environment variables or a `.env` file and do not commit credentials.

## My approach

I treat this repo as a live engineering notebook: exploring patterns, testing architectures, and building practical prototypes that sit between research and application. The emphasis is on learning by building, validating assumptions through working examples, and iterating toward more robust AI systems.

## Contact

- GitHub: https://github.com/ShivaUlavi
- Email: shivachetanulavi@gmail.com

---

This repository reflects my ongoing work in generative AI, agentic systems, RAG, multimodal reasoning, and applied LLM engineering.
