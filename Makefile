
# Default target
.DEFAULT_GOAL := help

# General
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# Project name
project_name := "deepresearch"
project_dir := "$(current_abs_path)"

# Environment variables
include .env

# Check required environment variables
ifeq ($(DATA_DIR),)
    $(error DATA_DIR must be set in .env file)
endif

# Global mount for data directory
mount_data := -v $(DATA_DIR):/project/data

.PHONY: help build run-interactive test \
        demo-research demo-react-cli research-workflow react-cli \
        load-data query-docs test-search \
        chainlit chainlit-direct \
        deploy-start deploy-logs stop clean

#
# Main Research Tools
#

demo-research: ## Run deep research workflow with HPC demo question
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.main "what are scientific applications of HPC?"

demo-react-cli: ## Run ReAct CLI with climate demo question
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.react_cli "What is the impact of global warming on ocean currents?"

research-workflow: ## Run research workflow with custom topic (usage: make research-workflow TOPIC="your topic")
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.main "$(TOPIC)"

react-cli: ## Run ReAct CLI with custom question (usage: make react-cli QUESTION="your question")
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.react_cli "$(QUESTION)"

#
# Data Management
#

load-data: ## Load sample papers into vector database
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.scripts.load_vector_db

query-docs: ## Query documents with custom question (usage: make query-docs QUERY="your question")
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.scripts.query_documents "$(QUERY)"

test-search: ## Test vector search functionality
	docker compose run --rm $(mount_data) $(project_name) uv run python -m deepresearch.scripts.test_vector_search

#
# Web Interface (Chainlit)
#

chainlit: start ## Start Chainlit frontend with llama-deploy backend
	docker compose run --rm $(mount_data) -p 8000:8000 $(project_name) uv run chainlit run src/deepresearch/chainlit_deploy_app.py

chainlit-direct: ## Start Chainlit frontend with direct workflow (starts qdrant)
	@docker compose up -d qdrant
	@sleep 2
	docker compose run --rm $(mount_data) -p 8000:8000 $(project_name) uv run chainlit run src/deepresearch/chainlit_direct_app.py

chainlit-react: ## Start Chainlit frontend with ReAct agent (starts qdrant)
	@docker compose up -d qdrant
	@sleep 2
	docker compose run --rm $(mount_data) -p 8000:8000 $(project_name) uv run chainlit run src/deepresearch/chainlit_react.py


#
# Development
#

build: ## Build Docker image
	docker compose build

run-interactive: build ## Run interactive bash session in container
	docker compose run -it --rm $(mount_data) $(project_name) /bin/bash

test: build ## Run all tests with pytest
	docker compose run --rm $(mount_data) $(project_name) uv run python -m pytest -v

logs: ## View service logs
	docker compose logs -f

start: ## Start all llama-deploy services
	docker compose up -d

stop: ## Stop all services
	docker compose down

clean: ## Clean up Docker images and containers
	docker compose down --rmi all --volumes --remove-orphans
	docker image prune -f

#
# Help
#

help: ## Show this help message
	@echo "Deep Research Assistant - Available Commands"
	@echo ""
	@echo "=== Quick Start ==="
	@echo "  make demo-research                  Run deep research workflow demo"
	@echo "  make demo-react-cli                 Run ReAct CLI demo"
	@echo "  make chainlit                       Start web interface"
	@echo ""
	@echo "=== Main Research Tools ==="
	@echo "  demo-research                       Run deep research workflow with HPC demo"
	@echo "  demo-react-cli                      Run ReAct CLI with climate demo"
	@echo "  research-workflow TOPIC=\"...\"       Run research workflow with custom topic"
	@echo "  react-cli QUESTION=\"...\"            Run ReAct CLI with custom question"
	@echo ""
	@echo "=== Data Management ==="
	@echo "  load-data                           Load sample papers into vector database"
	@echo "  query-docs QUERY=\"...\"              Query documents with custom question"
	@echo "  test-search                         Test vector search functionality"
	@echo ""
	@echo "=== Web Interface ==="
	@echo "  chainlit                            Start Chainlit with llama-deploy backend"
	@echo "  chainlit-direct                     Start Chainlit with direct workflow"
	@echo "  chainlit-react                      Start Chainlit with ReAct agent"
	@echo ""
	@echo "=== Production Deployment ==="
	@echo "  deploy-start                        Start all llama-deploy services"
	@echo "  stop                                Stop all services"
	@echo "  deploy-logs                         View llama-deploy service logs"
	@echo ""
	@echo "=== Development ==="
	@echo "  build                               Build Docker image"
	@echo "  run-interactive                     Run interactive bash session"
	@echo "  test                                Run all tests with pytest"
	@echo "  clean                               Clean up Docker images and containers"
	@echo ""
	@echo "=== Usage Examples ==="
	@echo "  make demo-research"
	@echo "  make research-workflow TOPIC=\"quantum computing\""
	@echo "  make react-cli QUESTION=\"What is CRISPR gene editing?\""
	@echo "  make query-docs QUERY=\"climate modeling\""
	@echo "  make chainlit                       # Then open http://localhost:8000"
	@echo ""
	@echo "=== Environment Variables (set in .env file) ==="
	@echo "  DATA_DIR          Path to data directory"
	@echo "  CEREBRAS_API_KEY  Cerebras API key for LLM"
	@echo "  OPENAI_API_KEY    OpenAI API key for embeddings"
	@echo "  TAVILY_API_KEY    Tavily API key for web search"
	@echo ""

