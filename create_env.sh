#!/bin/bash

#!/bin/bash
git submodule update --init --recursive --jobs 0

CONDA_PATH=~/anaconda3/

if test -f "$CONDA_PATH/etc/profile.d/conda.sh"; then
    echo "Found Conda at $CONDA_PATH"
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda --version
else
    echo "Could not find conda!"
fi



conda update -n base -c defaults conda

conda create -y -n neat python=3.8.1

conda activate neat

conda install -y ncurses=6.3 -c conda-forge
conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.3 cudatoolkit=11.3 -c nvidia -c conda-forge
conda install -y configargparse=1.4 astunparse=1.6.3 numpy=1.21.2 ninja=1.10.2 pyyaml mkl=2022.0.1 mkl-include=2022.0.1 setuptools=58.0.4 cmake=3.19.6 cffi=1.15.0 typing_extensions=4.1.1 future=0.18.2 six=1.16.0 requests=2.27.1 dataclasses=0.8
conda install -y magma-cuda110=2.5.2 -c pytorch
conda install -y -c conda-forge coin-or-cbc=2.10.5 glog=0.5.0 gflags=2.2.2 protobuf=3.13.0.1 freeimage=3.17 tensorboard=2.8.0

