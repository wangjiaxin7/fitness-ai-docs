FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libegl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mediapipe==0.10.35 \
    opencv-python-headless==4.13.0.92 \
    numpy==2.4.6 \
    matplotlib==3.10.9 \
    fastapi==0.136.3 \
    uvicorn \
    python-multipart

COPY analyze.py /app/analyze.py
COPY server.py /app/server.py

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
