FROM nvcr.io/nvidia/pytorch:24.02-py3
WORKDIR /app
COPY . .
RUN pip install -e .
