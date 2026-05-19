# The below code snippet is a pseudo-code, and it is not an actual working code.

import triton
import triton.language as tl

# Constants
BK = 1024
BV = 512

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, h_ptr,
    T, H, D, scale, i_h,
    USE_INITIAL_STATE, STORE_FINAL_STATE,
    initial_state_ptr=None, final_state_ptr=None
):
    # Kernel code goes here

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, h_ptr, do_ptr,
    T, H, D, scale, i_h,
    USE_INITIAL_STATE, STORE_FINAL_STATE,
    initial_state_ptr=None, final_state_ptr=None
):
    # Kernel code goes here

def fused_recurrent_retention(
    q, k, v, o, h=None,
    scale=1.0, i_h=0,
    USE_INITIAL_STATE=False, STORE_FINAL_STATE=False
):
    # Wrapper code goes here
