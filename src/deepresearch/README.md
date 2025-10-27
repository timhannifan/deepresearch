# Deep Research Agent - Week 2-3 Assignment

This directory contains the infrastructure for building a multi-agent deep research system.

## What's Provided

### Vector Database Infrastructure
**Ready to use** - No implementation needed

- `utils.py`: Shared Qdrant and embedding utilities
- `scripts/load_vector_db.py`: Script to load papers into Qdrant (with --overwrite option)
- `scripts/query_documents.py`: Script to query existing documents
- `scripts/test_vector_search.py`: Test vector search functionality
- `tools/vector_search.py`: Qdrant RAG search tool

- `scripts/test_vector_search.py`: Demonstration with multiple example queries

### Sample Data
**Included** - 3 sample scientific papers in `data/sample_papers/`

- Climate modeling
- Materials science  
- High-performance computing

## What You Need to Build

### 1. Web Search Tool (`tools/web_search.py`)
Implement a Tavily web search tool following the notebook pattern.

**Reference**: `notebooks/agentic_workflows.ipynb`, cell 7

### 2. Three Agents (`agents/`)
Create FunctionAgent instances for:
- `question_agent.py`: Generates research questions
- `answer_agent.py`: Answers using both tools
- `report_agent.py`: Synthesizes final report

**Reference**: `notebooks/agentic_workflows.ipynb`, cell 100

### 3. Workflow (`workflows/research_workflow.py`)
Build the orchestration workflow with these steps:
- Setup: Initialize agents from StartEvent
- Generate: Create research questions
- Answer: Parallel answering of questions
- Report: Synthesize final output

**Reference**: `notebooks/agentic_workflows.ipynb`, cells 101-102

### 4. CLI Interface (`main.py`)
Create command-line interface to run the workflow.

**Usage**: `uv run python -m deepresearch.main "Your topic"`

## Quick Start

### Prerequisites
Make sure you have the services running:
```bash
# Terminal 1: Start services (proxy + qdrant)
make run-services

# Terminal 2: Start interactive container
make run-interactive
```

### Load and Query Data

1. **Load sample data** (one-time setup):
   ```bash
   # Safe mode - won't overwrite existing collection
   uv run python -m deepresearch.scripts.load_vector_db
   
   # Or overwrite existing data
   uv run python -m deepresearch.scripts.load_vector_db --overwrite
   ```

2. **Query documents**:
   ```bash
   # Single query
   uv run python -m deepresearch.scripts.query_documents "What is climate modeling?"
   
   # With custom top-k
   uv run python -m deepresearch.scripts.query_documents "HPC benefits" --top-k 5
   ```

3. **Test and demo vector search**:
   ```bash
   # Run demonstration with multiple example queries
   uv run python -m deepresearch.scripts.test_vector_search
   
   # Or test a specific query
   uv run python -m deepresearch.scripts.test_vector_search "climate models"
   
   # With custom top-k
   uv run python -m deepresearch.scripts.test_vector_search "HPC" --top-k 5
   ```

### Build Your Implementation

4. **Build your implementation**:
   - Start with `tools/web_search.py`
   - Create agents in `agents/`
   - Build workflow in `workflows/`
   - Add CLI in `main.py`

5. **Test end-to-end**:
   ```bash
   uv run python -m deepresearch.main "Climate modeling"
   ```

## Architecture

```
User Input (topic)
    ↓
StartEvent → QuestionAgent
    ↓
GenerateEvent → [QuestionEvent, QuestionEvent, ...]
    ↓
Multiple AnswerAgents (parallel)
    ↓
[AnswerEvent, AnswerEvent, ...] → ReportAgent
    ↓
StopEvent → Final Report
```

## Success Checklist

- [ ] Web search tool implemented and working
- [ ] All three agents created with proper system prompts
- [ ] Workflow orchestrates agents correctly
- [ ] Parallel question answering works
- [ ] Progress events stream during execution
- [ ] Final report is comprehensive and well-formatted
- [ ] Passes `ruff check src/deepresearch/`
- [ ] Can run: `uv run python -m deepresearch.main "Any topic"`
