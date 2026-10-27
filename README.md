# GenAI: Generative AI, Agentic Systems, and Multimodal Experiments

This repository is a hands-on portfolio of generative AI and agentic workflow work built around LangChain, LangGraph, retrieval-augmented generation (RAG), structured tool use, and multimodal AI. It brings together notebooks, demos, and prototype apps that explore how LLM-based systems can reason, retrieve information, generate structured output, and interact with external tools.

The work in this repo reflects an ongoing learning and experimentation path across several core themes:

- Agentic orchestration with LangGraph
- Retrieval-augmented generation and document QA
- Tool calling, planning, and structured outputs
- Multimodal workflows for vision and video analysis
- Local model experimentation and API token testing
- Validation loops for code generation and workflow reliability

## Current focus areas

### 1. LangGraph and agentic orchestration
The repository includes graph-based examples for building stateful, tool-using, and multi-step AI systems. Notable work includes:

- Simple LangGraph demos and routing patterns
- Graphs with output parsing and structure enforcement
- Multi-agent architecture experiments for collaborative reasoning
- Basic chatbot patterns and conversational orchestration

### 2. Retrieval-Augmented Generation (RAG)
A major part of the repo focuses on building practical RAG systems for document processing and knowledge retrieval:

- Document loading and chunking workflows
- Retriever implementations for query-to-context matching
- LLM integration for grounded generation
- Streamlit-based RAG applications for corporate or domain-specific Q&A
- Citation-aware retrieval workflows for documentation use cases

### 3. Tool use and agent capabilities
Several examples explore how agents can call tools, execute plans, and return structured results:

- Built-in tool usage demos
- Custom tools for domain-specific integrations
- Tool node patterns in agent graphs
- Plan-and-solve workflows
- Structured output schemas for deterministic agent responses

### 4. Multimodal AI and video understanding
The repository includes a growing set of multimodal experiments focused on visual understanding and fitness/video analysis:

- Sequential multimodal narrative generation
- Asynchronous multimodal evaluation pipelines
- Local multimodal analysis using Ollama and vision-capable models
- Batch video analysis using parallel processing
- WorkoutAnalyzerApp for video upload and technique feedback

### 5. Software and data analysis agents
The repo also includes experimentation with agents for technical tasks and analytical workflows:

- RAG-based code generation with validation loops
- Python code generation patterns
- Agent workflows aimed at software engineering and analytical decision support

### 6. Prompting, error handling, and model testing
Supporting notebooks and scripts focus on reliability and experimentation:

- Prompt engineering templates
- Error handling and retry patterns
- Local model inference checks
- API token validation for OpenAI, Claude, Gemini, and Hugging Face
- Ollama and local LLM testing workflows

## Repository structure

- `LangGraphDemo/` — LangGraph workflows, graph orchestration, and multi-agent demos
- `RAG/` — retrieval pipelines, retrievers, LLM wrappers, and Streamlit app prototypes
- `Tools/` — tool integration demos, output schemas, and planning examples
- `MultiModalAgents/` — multimodal and video-analysis experiments
- `SoftwareAndDataAnalysisAgents/` — code-generation and analysis-agent prototypes
- `TestScripts/` — local model validation and provider token tests
- `Prompt_templates.ipynb` — structured prompting experiments
- `ErrorHandling.ipynb` — retry, recovery, and failure-handling patterns
- `requirements.txt` — Python dependencies used across the project

## Notable examples

- `RAG/streamlit_app_V3.py` — interactive RAG app for document-based Q&A
- `RAG/rag.py` — orchestration of the retrieval + generation pipeline
- `MultiModalAgents/04_Batch_Video_Analysis_ThreadPool.py` — parallelized video analysis workflow
- `MultiModalAgents/WorkoutAnalyzerApp/` — app-based workout feedback and form assessment
- `Tools/CustomToolsDemo.ipynb` — custom tool orchestration examples
- `Tools/PlanAndSolve.ipynb` — planning-first agent workflows
- `SoftwareAndDataAnalysisAgents/RAG_Code_Generation_with_Validation_Loop.ipynb` — reasoning + validation loop for code generation

## Getting started

1. Clone the repository

```bash
git clone https://github.com/ShivaUlavi/GenAI.git
cd GenAI
```

2. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Launch notebooks or apps as needed

```bash
jupyter lab
```

For app-based demos, use the relevant script or Streamlit app in the directory you want to run.

## Environment variables and setup notes

Some demos rely on external APIs or local services. Common environment variables and services used across examples include:

- `AICREDITS_API_KEY`
- `AICREDITS_BASE_URL`
- `HF_TOKEN`
- `TAVILY_API_KEY`
- Local Ollama instances for vision/local model workflows

Keep secrets in environment variables or a local `.env` file and avoid committing credentials.

## Status

This repo is an active learning and prototyping workspace. It contains experiments, research notebooks, and proof-of-concept systems rather than a single production-ready product. The focus is on exploring architectures, workflows, and practical patterns in modern generative AI.

## Contact

GitHub: https://github.com/ShivaUlavi
Email: shivachetanulavi@gmail.com

---

This repository documents the progression of work from foundational LLM workflows to agentic systems, multimodal reasoning, and retrieval-first AI applications.
