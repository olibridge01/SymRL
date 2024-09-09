FROM ubuntu:18.04
# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get update --fix-missing && apt-get install -y curl grep sed zip unzip dpkg netcat iproute2 dnsutils vim && \
    TINI_VERSION=0.19.0 && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN apt-get update && apt-get install -y build-essential cmake sqlite3 libsqlite3-dev zip unzip \
    libboost-dev libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-python-dev libboost-regex-dev libboost-numpy-dev

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

SHELL ["/bin/bash", "-c"]

WORKDIR /relnet
ADD ./environment.yml /relnet/environment.yml
RUN conda env update --file environment.yml

ENV PATH /opt/conda/envs/symrl-cenv/bin:$PATH
ENV CPLUS_INCLUDE_PATH $CPLUS_INCLUDE_PATH:/usr/include/python3.6/

ADD ./relnet/objective_functions /relnet/relnet/objective_functions

RUN source activate symrl-cenv && cd /relnet/relnet/objective_functions && make

RUN source activate symrl-cenv && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.2.html
RUN source activate symrl-cenv && pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.2.html
RUN source activate symrl-cenv && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.2.html
RUN source activate symrl-cenv && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.2.html

RUN mkdir -p /usr/local/relnet/
RUN cp /relnet/relnet/objective_functions/objective_functions_ext.so /usr/local/relnet
ENV PYTHONPATH /usr/local/relnet:$PYTHONPATH

CMD ["bash", "/relnet/docker/psn/start_psn.sh"]