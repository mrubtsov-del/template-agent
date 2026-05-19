FROM registry.access.redhat.com/ubi9/python-312:latest

# --------------------------------------------------------------------------------------------------
# set the working directory to /app
# --------------------------------------------------------------------------------------------------

WORKDIR /app

# --------------------------------------------------------------------------------------------------
# Copy manifest files and install python packages
# --------------------------------------------------------------------------------------------------

USER root
COPY pyproject.toml /app/pyproject.toml
RUN pip install uv
ENV UV_HTTP_TIMEOUT=180 \
    UV_CONCURRENCY=2 \
    PIP_NO_CACHE_DIR=1
RUN uv venv
RUN uv pip install --no-cache -r pyproject.toml
USER default

# --------------------------------------------------------------------------------------------------
# copy source code and files
# --------------------------------------------------------------------------------------------------

COPY template_agent /app/template_agent
COPY vendor /app/vendor

# --------------------------------------------------------------------------------------------------
# Set PYTHONPATH to include /app
# --------------------------------------------------------------------------------------------------

ENV PYTHONPATH=/app:/app/vendor


# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

CMD ["/app/.venv/bin/python", "-m", "template_agent.src.main"]
