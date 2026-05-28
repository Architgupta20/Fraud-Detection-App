FROM openjdk:17-jdk-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "Outputs/Reports/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
