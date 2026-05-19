import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    X, R, out,
    xnumel, rnumel,
    in_out_ptr0, in_out_ptr1,
    in_ptr1, in_ptr2,
    out_ptr0,
    **meta):
    ...
