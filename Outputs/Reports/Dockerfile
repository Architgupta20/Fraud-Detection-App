# Use base image that already has Java (for Spark) and a clean Linux environment
FROM eclipse-temurin:17-jdk-jammy

# Install Python 3 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Allow pip to install packages system-wide (fixes PEP 668 issue)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY Outputs/Reports/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy entire project into the container
COPY . /app

# Environment settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit from its real project path (keeps config.py import working)
CMD ["streamlit", "run", "Outputs/Reports/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
