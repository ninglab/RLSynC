FROM python:3.8.13

WORKDIR /
RUN mkdir -p /rlsync
COPY requirements.txt /rlsync/requirements.txt
WORKDIR /rlsync
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
COPY submodules ./submodules
COPY rlsync ./rlsync
COPY pyproject.toml .
RUN cd submodules/MolecularTransformer && pip install -e . && cd /rlsync && pip install -e .
COPY scripts ./scripts
CMD ["/bin/bash"]