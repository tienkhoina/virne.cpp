FROM python:3.10.20-slim@sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements-nonml.txt /tmp/requirements-nonml.txt
RUN python -m pip install \
    --no-cache-dir \
    --only-binary=:all: \
    --requirement /tmp/requirements-nonml.txt

CMD ["python", "--version"]
