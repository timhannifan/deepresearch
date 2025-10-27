# deepresearch

An AI-powered research assistant that conducts comprehensive research using intelligent agents, vector search, and web research capabilities.

## Quick Start

### CLI Research Tools

Run research workflows directly from the command line:

```bash
# Research a topic using the deep research workflow
make research-workflow TOPIC="climate change impacts"

# Or use the ReAct agent with a specific question
make react-cli QUESTION="What are the latest advances in CRISPR gene editing?"

# Demo examples
make demo-research        # HPC applications demo
make demo-react-cli       # Climate change demo
```

### Web Interface

Choose from three different web interfaces:

```bash
# 1. Chainlit with llama-deploy backend (production)
make chainlit

# 2. Chainlit with direct workflow (development)
make chainlit-direct

# 3. Chainlit with ReAct agent (interactive reasoning)
make chainlit-react
```

Then open your browser to http://localhost:8000

## Available Tools

### Main Research Tools

- **`make research-workflow TOPIC="..."`** - Run deep research workflow on a custom topic
- **`make react-cli QUESTION="..."`** - Run ReAct agent with a custom question

### Data Management

- **`make load-data`** - Load sample papers into vector database
- **`make query-docs QUERY="..."`** - Query documents with a custom question
- **`make test-search`** - Test vector search functionality

### Web Interface

- **`make chainlit`** - Start Chainlit with llama-deploy backend
- **`make chainlit-direct`** - Start Chainlit with direct workflow (auto-starts qdrant)
- **`make chainlit-react`** - Start Chainlit with ReAct agent (auto-starts qdrant)

### Production Deployment

- **`make start`** - Start all llama-deploy services
- **`make stop`** - Stop all services
- **`make logs`** - View service logs

### Development

- **`make build`** - Build Docker image
- **`make run-interactive`** - Run interactive bash session in container
- **`make test`** - Run all tests with pytest
- **`make clean`** - Clean up Docker images and containers
- **`make help`** - Show all available commands

## Environment Setup

Create a `.env` file in the project root:

```env
DATA_DIR=/path/to/your/data/directory
OPENAI_API_KEY=your_openai_api_key
CEREBRAS_API_KEY=your_cerebras_api_key
TAVILY_API_KEY=your_tavily_api_key
```