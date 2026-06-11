FROM analyze-demo:latest AS base
RUN pip install -q --no-cache-dir fastapi uvicorn python-multipart 2>&1 | tail -1
COPY analyze.py /app/analyze.py
COPY server.py /app/server.py
WORKDIR /app
ENTRYPOINT []
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
