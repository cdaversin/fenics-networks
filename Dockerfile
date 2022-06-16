FROM ceciledc/fenics_mixed_dimensional:latest
USER root
RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get clean && \
    apt-get -y install python3-h5py && \
    pip install --upgrade pip && \
    pip install meshio[all] && \
    pip install networkx && \
    pip install pandas && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER root
