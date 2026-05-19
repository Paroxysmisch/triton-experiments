import triton
import numpy as np

def index_select_cat_bwd(grad_source, grad_output, index):
    assert grad_source.ndim == 2 and grad_source.device.type() == 'cuda'
    assert grad_output.ndim == 2 and grad_output.device.type() == 'cuda'
    assert index.ndim == 2 and index.device.type() == 'cuda'
    assert grad_source.shape[1] == grad_output.shape[1]
    assert grad_source.shape[0] == index.shape[0]
    assert index.shape[1] == grad_output.shape[0]

    B, C, N = grad_source.shape[0], grad_source.shape[1], index.shape[1]
    grid = lambda M: triton.next_power_of_2(M) // 32 * 32
    triton.launch(
        index_select_cat_bwd_kernel,
        (grid(B * C),),
        (B, C, N, grad_source.device.ptr(), grad_output.device.ptr(), index.device.ptr()),
        device=grad_source.device,
    )
