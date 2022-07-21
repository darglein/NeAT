#!/bin/bash

CONDA_PATH=~/anaconda3/

if test -f "$CONDA_PATH/etc/profile.d/conda.sh"; then
    echo "Found Conda at $CONDA_PATH"
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda --version
else
    echo "Could not find conda!"
fi


conda activate neat


#cd External/pytorch

mkdir External/
cd External/


wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip -O  libtorch.zip
unzip libtorch.zip -d .
#cd libtorch

cp -rv libtorch/ $CONDA_PATH/envs/neat/lib/python3.8/site-packages/torch/




