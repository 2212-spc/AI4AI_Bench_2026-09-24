FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /srv/lab
COPY engine.py e1_world.py lab_service.py world.json /srv/lab/
RUN mkdir -p /var/lab
EXPOSE 18777
CMD ["python3", "/srv/lab/lab_service.py"]
