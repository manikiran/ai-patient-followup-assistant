FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the local embedding model (all-MiniLM-L6-v2, ~80 MB) into the image so
# the container never downloads it at runtime.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

COPY schema/ ./schema/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/patients/ ./data/patients/
COPY app.py .

# No secrets in the image: LLM settings (and GOOGLE_API_KEY if using Gemini)
# and S3_BUCKET are passed at run time; S3 access comes from the EC2 IAM role.
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
