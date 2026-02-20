FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy uv lock and pyproject.toml
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Place the virtual environment in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
