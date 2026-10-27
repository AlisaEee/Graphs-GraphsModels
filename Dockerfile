FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY graph_chat /app/graph_chat
COPY graph_preprocess/C3_A4_posweight_model.pt /app/data/C3_A4_posweight_model.pt
COPY graph_preprocess/graphrag_preprocessed_rubert/output/entities.parquet /app/data/graphrag_preprocessed_rubert/output/entities.parquet
COPY graph_preprocess/graphrag_preprocessed_rubert/output/relationships.parquet /app/data/graphrag_preprocessed_rubert/output/relationships.parquet

EXPOSE 8501
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "/app/graph_chat/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
