# Dockerfile - assumes streamlit_app.py, requirements.txt etc are at repo root
FROM openjdk:17-jdk-slim

# Install Python and basic tools (no heavy extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (repo root)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy repository content into container
COPY . /app

# Ensure streamlit app is at /app root (your file is already at root)
# This line is harmless if file is already in place
RUN cp /app/streamlit_app.py /app/streamlit_app.py || true

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
