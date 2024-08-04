#!/usr/bin/env python
# coding: utf-8

import os
import torch

def check_versions():
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA version: {torch.version.cuda}')

def enable_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

def disable_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
