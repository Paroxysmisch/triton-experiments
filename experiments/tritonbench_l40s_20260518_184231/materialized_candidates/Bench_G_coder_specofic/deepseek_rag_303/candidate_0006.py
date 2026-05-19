import triton
import triton.language as tl

try:
    from mamba_ssd.utils import wrap_kernel_launcher
except ImportError:
    from deepspeed.accelerator import wrap_kernel_launcher

try:
    from mamba_ssd.utils import CUDA_CAPABILITY
except ImportError:
    from deepspeed.accelerator import CUDA_CAPABILITY

try:
    from mamba_ssd.ops.triton_ops.patch_ops import _fwd_kernel as _fwd_kernel_aligned
except ImportError:
    from deepspeed.ops.triton_ops.patch import _fwd_kernel as _fwd_kernel_aligned


def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, B0, V, Out, B_Start_Loc, B_Seqlen, MAX_INPUT_LEN):
    Lq, Lk, Lb0, Lv = Q.shape[-1], K.shape[-1], B0.shape[-1], V.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lb0 == 128
    assert Lk in {16, 32, 64, 128, 256}

    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = B_Seqlen.shape[0], Q.shape[1]

    grid = (batch, head, triton.cdiv(MAX_INPUT_LEN, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel_aligned[grid](
        Q,
        K,
        B0,
        V,
        sm_scale,
        B_Start_Loc,
        B_Seqlen,
        Out,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=1,
    )


# Pre-compile the kernel with INT8 tensors
if torch.cuda.get_autocast_mode() == torch.cuda.AutocastMode.ENABLETED:
    q = torch.empty((2, 2, 128), dtype=torch.float16, device="cuda")
    k = torch.empty((2, 2, 128), dtype=torch.float16, device="cuda")
    v = torch.empty((2, 2, 128), dtype=torch.float16, device="cuda")
    b0 = torch.empty((2, 2, 128), dtype=torch.bfloat16, device="cuda")
    o = torch.empty((2, 2, 128), dtype=torch.bfloat16, device="cuda")
    b_start_loc = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    b_seq_len = torch.tensor([1, 1], dtype=torch.long, device="cuda")
    max_len = 2
    _attention_rel_h_rel_w_kernel_aligned_device(
        q, k, b0, v, o, b_start_loc, b_seq_len, max_len
    )


@contextmanager
def autocast(enabled):
    if enabled:
        torch._C.set_prim_tensor_dtype_autocast_enabled(True)
        try:
            yield
        finally:
            torch._C.set_prim_tensor_dtype_autocast_enabled(False)
    else:
        yield
