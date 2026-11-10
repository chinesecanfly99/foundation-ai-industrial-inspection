FROM python:3.11-slim AS assets
WORKDIR /build
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir onnx==1.17.0 numpy==2.2.6
COPY classroom_demo/fetch_assets.py classroom_demo/cloud_export.py ./
RUN python fetch_assets.py && python cloud_export.py torch_cache/hub/checkpoints/resnet18-f37072fd.pth feature_backbone.onnx

FROM python:3.11-slim
WORKDIR /srv/inspection
ENV PYTHONPATH=/srv/inspection/src:/srv/inspection PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False INSPECTION_CLOUD_EDITION=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir 'gradio==5.49.1' 'gradio_client==1.13.3' 'pydantic==2.11.10' 'fastapi==0.116.1' 'starlette==0.47.3' 'uvicorn==0.35.0' 'anyio==4.10.0' 'python-multipart==0.0.20' 'numpy==2.2.6' 'pandas==2.3.3' 'Pillow==11.3.0' 'opencv-python-headless==4.12.0.88'
COPY app ./app
COPY src ./src
COPY configs ./configs
COPY data/manifests ./data/manifests
COPY samples ./samples
COPY classroom_demo/cloud_service.py classroom_demo/cloud_models.py ./classroom_demo/
COPY --from=assets /build/feature_backbone.onnx ./
COPY --from=assets /build/models ./models
RUN mkdir .cache && chown 10001 .cache
USER 10001
CMD ["sh","-c","exec uvicorn classroom_demo.cloud_service:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 40 --no-access-log"]
