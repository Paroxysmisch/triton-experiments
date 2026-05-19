import torch
import triton
import triton.language as tl

@triton.jit
def index_fill_(self, dim, index, value):
    return tl.device_index_fill_(self, dim, index, value)

def wrapper_func(self, dim, index, value):
    return index_fill_(self, dim, index, value)
