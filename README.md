## Install Instructions

* Prepare Host System (Ubuntu)
```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install g++-9
g++-9 --version # Should Print Version 9.4.0 or higher
```
* Create Conda Environment

```shell
./create_env.sh
```

* Install Pytorch

 ```shell
./install_pytorch_precompiled.sh
 ```

* Compile NeAT

```shell
conda activate neat

export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.8/site-packages/torch/;${CONDA}" ..
make -j10

```

## Run Instructions

* Get Pepper dataset from here: https://repository.kaust.edu.sa/handle/10754/676019
* Extract datasets
* Update the `main()` of `nikon2neat.cpp` to point to the downloaded dataset directory (the output should be into NeAT/scenes)
* Preprocess data using our nikon2neat programm:
 ```shell
mkdir scenes
cd NeAT
export LD_LIBRARY_PATH=~/anaconda3/envs/neat/lib
./build/bin/nikon2neat
 ```
* Update configuration file in configs/
* Run reconstruction
 ```shell
cd NeAT
export LD_LIBRARY_PATH=~/anaconda3/envs/neat/lib
./build/bin/reconstruct configs/pepper.ini
 ```
* The result will be written to NeAT/Experiments
* Use tensorboard for easy visualization:
 ```shell
conda activate neat
cd NeAT
tensorboard --logdir Experiments/ --samples_per_plugin images=100
 ```