# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory for config
RUN mkdir -p .streamlit

# Create Streamlit config for Cloud Run
RUN echo '\
[server]\n\
port = 8080\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = "#00d4ff"\n\
backgroundColor = "#0a0a0a"\n\
secondaryBackgroundColor = "#1a1a2e"\n\
textColor = "#ffffff"\n\
font = "sans serif"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > .streamlit/config.toml

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
