# Imagen para Hugging Face Spaces (SDK Docker). También sirve para correrla localmente:
#   docker build -t bonefracture-ai .
#   docker run -p 7860:7860 -e HF_WEIGHTS_REPO=<usuario>/bonefracture-ai-weights bonefracture-ai
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Hugging Face Spaces ejecuta el contenedor con el usuario UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# PyTorch solo CPU (mucho más liviano que la versión con CUDA)
RUN pip install --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# Un solo worker: cada worker carga los 3 modelos en memoria
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:7860", "run:app"]
