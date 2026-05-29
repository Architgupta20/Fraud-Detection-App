FROM python:3.12-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1
ENV BASE_DIR=/app
ENV MODEL_DATA_DIR=/app/Data/Model_Data
ENV SKLEARN_MODEL_PATH=/app/Models/gbt_sklearn.pkl

EXPOSE 8501
CMD ["streamlit", "run", "Outputs/Reports/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
