# This is a basic docker image for use in the clinic
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Switch to root to update and install tools
RUN apt-get update && apt-get install -y curl git

# Create working directory
WORKDIR /project

COPY pyproject.toml .
COPY uv.lock .

# Set UV to use copy mode to avoid hardlink warnings
ENV UV_LINK_MODE=copy

# Resolve and install Python packages from pyproject/uv.lock
RUN /usr/local/bin/uv venv
ENV VIRTUAL_ENV=/project/.venv
ENV PATH="/project/.venv/bin:$PATH"
ENV PYTHONPATH=/project/src
RUN uv sync

CMD ["/bin/bash"]