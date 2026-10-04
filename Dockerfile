# Web app in demo mode (see requirements-web.txt). Hosts such as Render set PORT; 8000 otherwise.
FROM python:3.11-slim

WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app.py enhance.py model.py ./
COPY templates templates
COPY static static

ENV PYTHONUNBUFFERED=1
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 app:app
