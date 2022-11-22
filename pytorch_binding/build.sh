#!/bin/bash

# TMPDIR=$PWD pip install torch==1.11.0  -i https://pypi.tuna.tsinghua.edu.cn/simple

export CUDA_HOME="/usr/local/cuda"
export C_INCLUDE_PATH="/usr/local/cuda/include"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include"

#python setup.py install
pip install -e 
