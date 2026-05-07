FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Build context = project root (see docker-compose.yml).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY app_streamlit/app.py app_streamlit/config.py app_streamlit/utils.py \
     app_streamlit/train_model.py app_streamlit/__init__.py ./
COPY app_streamlit/models ./models
COPY data ./data

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
