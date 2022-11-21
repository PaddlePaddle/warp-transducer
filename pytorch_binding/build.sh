#!/bin/bash

# TMPDIR=$PWD pip install torch==1.11.0  -i https://pypi.tuna.tsinghua.edu.cn/simple

export CUDA_HOME="/usr/local/cuda"

python setup.py install
