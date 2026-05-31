# Pointbench

Squared distance benchmark between a query point and a database of points.

## Build

### With Docker (recommended)

```bash
docker build -t pointbench .
docker run --rm --gpus all -it pointbench
```

Or with docker-compose:

```bash
docker-compose up --build
docker-compose run --rm pointbench
```

### Without Docker

```bash
mkdir build && cd build
cmake ..
make
./pointbench
```

## Run

Inside the container (or build directory):

```bash
./pointbench
```

## Requirements

- CMake 3.18+
- C++17 compiler
- CUDA-capable GPU (for CUDA benchmark)
- Docker with NVIDIA container runtime (for Docker build)