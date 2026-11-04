FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY evolve/ ./evolve/
COPY config/ ./config/
RUN python -m pip install --no-cache-dir '.[train]'
# No contact information or secret is baked into the image.
CMD ["python", "-m", "evolve", "daemon"]
