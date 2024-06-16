FROM ubuntu:20.04

# Set environment variables for locale
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    g++-9 \
    gcc-9 \
    curl \
    unzip \
    libopencv-dev \
    python3-dev \
    python3-pip &&   \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CXX=g++-9
ENV CXX_STANDARD=17

# Install Python packages
RUN pip3 install numpy matplotlib

# Clone libtorch
RUN mkdir /opt/libtorch && \
    cd /opt/libtorch && \
    curl -L -o libtorch.zip https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip

# Set environment variables for libtorch
ENV LIBTORCH_PATH=/opt/libtorch/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib:$LD_LIBRARY_PATH

# Clone matplotlib-cpp
RUN mkdir /opt/matplotlib-cpp && \
    cd /opt/matplotlib-cpp && \
    git clone https://github.com/lava/matplotlib-cpp.git

# Set the working directory in the container
WORKDIR /code

# Copy local files under the current dir into the container
COPY . /code

# Set the default command to bash
CMD ["/bin/bash"]
