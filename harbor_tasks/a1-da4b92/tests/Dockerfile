FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6
RUN mkdir -p /app /logs/verifier /tests
COPY a1_verify_core.py verify_a1.py verify_spec.json test.sh /tests/
