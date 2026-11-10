FROM python:3.11-slim
WORKDIR /srv/inspection
ENV PYTHONPATH=/srv/inspection/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/srv/inspection/torch_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir 'fastapi==0.116.1' 'uvicorn==0.35.0' 'numpy==2.2.6' 'Pillow==11.3.0' 'opencv-python-headless==4.12.0.88'
COPY src ./src
COPY configs/quality_rules.json configs/quality_rules_tile.json ./configs/
COPY samples/tile/crack_000.png ./samples/tile/crack_000.png
COPY samples/bottle/broken_large_000.png ./samples/bottle/broken_large_000.png
COPY app/localization.py ./app/localization.py
COPY app/locales/en.json ./app/locales/en.json
COPY classroom_demo/public_app.py classroom_demo/index.html classroom_demo/fetch_assets.py ./
RUN python fetch_assets.py
USER 10001
CMD ["sh","-c","exec uvicorn public_app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 16"]
