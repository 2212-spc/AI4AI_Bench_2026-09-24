FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6
RUN mkdir -p /app /logs/verifier /tests
COPY verify_o.py gen_inputs.py spec.json expected.npz test.sh /tests/
