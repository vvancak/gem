Graph Embedding Methods
-----------------------
Implementation part of the diploma thesis **Graph Data Analysis Using DeepLearning Methods**

Department of Software Engineering, MFF UK  
Supervised by  RNDr.  Martin Svoboda, Ph.D.

## Download & Setup
Execute the following steps in BASH / GIT BASH / WSL (untested):

##### 0. Requirements:
* ***git*** - [download here](https://git-scm.com/downloads)

##### 1. Clone this repository: 
```Shell
git clone https://github.com/vvancak/gem.git
```

##### 2. Project setup and the file structure:
```Shell
cd ./gem
chmod u+x install.sh
./install.sh #[optional: arguments]
```

## Install & Run
* The project can run directly on your PC without virtualization, or via Docker. 

* Due to the complexity of CUDA installation, the GPU support is available only within Docker.

* The following scripts should be executed from the project root.

-----------------------
Execute the following steps in BASH / GIT BASH / WSL (untested):

### a) Pipenv [CPU]
##### 0. Install Requirements:
* ***python*** v3.6 with ***pip*** - [download here](https://www.python.org/downloads/)
* ***pipenv*** - [install here](https://docs.pipenv.org/en/latest/install/#installing-pipenv)

##### 1. Install depenencies:
```Shell
pipenv install --skip-lock 
```

##### 2. Run the project environment
```Shell
pipenv shell
```

##### 3. Launch the project
```Shell
cd ./src
python3 ./main.py #[optional: arguments or -h for help]
```

### b) Docker [CPU]
##### 0. Install Requirements:
* ***docker*** - [download here](https://www.docker.com/)

##### 1. Install depenencies:
```Shell
cd ./cpu-docker
docker build -t gem-docker-cpu .
cd ..
```

##### 2. Run the project environment
```Shell
docker run --rm -it \
    -v $(pwd):/usr/local/gem \
    -u $(id -u):$(id -g) \
    -w /usr/local/gem \
    gem-docker-cpu bash
```

##### 3. Launch the project
```Shell
cd /usr/local/gem/src
python3 ./main.py #[optional: arguments or -h for help]
```


### c) Docker [GPU]

##### 0. Install Requirements:
* ***nvidia-docker*** - [download here](https://github.com/NVIDIA/nvidia-docker)

##### 1. Install depenencies:
```Shell
cd ./gpu-docker
docker build -t gem-docker-gpu .
cd ..
```

##### 2. Run the project environment
One-GPU Server:
```Shell
nvidia-docker run --rm -it \
    -v $(pwd):/usr/local/gem \
    -u $(id -u):$(id -g) \
    -w /usr/local/gem \
    gem-docker-gpu bash
```

Multiple-GPU Server:
```Shell
GPU=0 # or replace 0 with an available GPU
NV_GPU=${GPU} nvidia-docker run --rm -it \
    -v $(pwd):/usr/local/gem \
    -u $(id -u):$(id -g) \
    -w /usr/local/gem \
    gem-docker-gpu bash
```


##### 3. Launch the project
```Shell
cd /usr/local/gem/src
python3 ./main.py #[optional: arguments or -h for help]
```