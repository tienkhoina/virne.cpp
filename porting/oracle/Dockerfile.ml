FROM virne-python-oracle:py310-nonml

# Share the non-ML layers. CPU wheels avoid downloading CUDA runtimes;
# production LibTorch stays in the bind-mounted workspace, outside this image.
RUN python -m pip install --only-binary=:all: pip==25.1.1 \
 && python -m pip install --only-binary=:all: \
        torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m pip install --only-binary=:all: \
        --requirement /tmp/requirements-ml.txt \
 && python -m pip install --only-binary=:all: --no-index --no-deps \
        --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html \
        torch_scatter==2.1.2+pt26cpu torch_sparse==0.6.18+pt26cpu \
 && python -m pip check \
 && python -c "import torch, torch_geometric, torch_scatter, torch_sparse; assert torch.__version__ == '2.6.0+cpu'; assert torch.version.cuda is None; assert torch_geometric.__version__ == '2.7.0'"

WORKDIR /work
CMD ["python", "--version"]
