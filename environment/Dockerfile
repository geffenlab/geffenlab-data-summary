# This runs in an Ubuntu environment.
FROM ubuntu:22.04

# Install wget and https certificate info, which we'll use to install things, below.
# Clean up cache and temp files along the way.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install --no-install-recommends --yes wget ca-certificates \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install conda.
RUN mkdir -p /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \
    && bash /opt/conda/miniconda.sh -b -p /opt/miniconda

# Create our conda environment from environment.yml.
COPY environment/environment.yml /opt/environment/environment.yml
RUN /opt/miniconda/bin/conda env create -f /opt/environment/environment.yml

COPY code/*.py /opt/code/
ENTRYPOINT ["/opt/miniconda/bin/conda", "run", "--no-capture-output", "-n", "geffenlab-data-summary", "python", "/opt/code/run_data_collection.py"]
