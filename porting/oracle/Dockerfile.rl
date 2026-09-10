FROM virne-python-oracle:py310-torch26-pyg27-cpu

# Only dependencies needed by the original environment/network classes.
# Reuse all Torch/PyG layers; production C++ has no Python runtime dependency.
RUN python -m pip install --no-cache-dir \
        omegaconf==2.3.0 gym==0.22.0 pandas==2.2.3 \
 && python -m pip check

WORKDIR /work
