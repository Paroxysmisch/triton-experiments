import torch
import triton
import triton.language as tl
from .triton_utils.kernels import rms_norm, silu
from .utils import is_fp8_available

if is_fp8_available():
    @triton.jit
    def ff_llama(x, w1, rms_w, w3, o, scale, n_cols: int, n_ctx: int, eps=1e-6, BLOCK_SIZE_M: tl.constexpr = 256, BLOCK_SIZE_N: tl.constexpr = 256, BLOCK_SIZE_K: tl.constexpr = 64, IS_FP8: tl.constexpr = True):
        # Kernel for computing F.silu(w1(x)) * w3(x) with RMS norm.
        range_m = tl.arange(0, BLOCK_SIZE_M)
        range_n = tl.arange(0, BLOCK_SIZE_N)
        rms_range = tl.arange(0, BLOCK_SIZE_K)
        m_mask = range_m[:, None] < n_ctx
        n_mask = range_n[None, :] < n_cols

        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        x_ptr = x + range_m * n_cols + range_n
        w1_ptr = w1 + range_n * n_cols + range_m
        rms_w_ptr = rms_w + rms_range
        w3_ptr = w3 + range_m * n_cols + range_n
        o_ptr = o + range_m * n_cols + range_n

        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE_K)):
            k_mask = (rms_range < n_cols) & (rms_range >= i * BLOCK_SIZE_K)
            rms_w_val = tl.load(rms_w_ptr + i * BLOCK_SIZE_K, mask=k_mask, other=1.0).to(tl.float32)
            x_val = tl.load(x_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            w1_val = tl.load(w1_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            w3_val = tl.load(w3_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)

            x_val = rms_norm(x_val, rms_w_val)
            acc1 += tl.dot(x_val, w1_val, allow_tf32=False)
            acc2 += tl.dot(w1_val, x_val, allow_tf32=False)
            tl.store(o_ptr, (silu(acc1) * acc2).to(o_ptr.dtype.element_ty), mask=m_mask & n_mask)
            x_ptr += BLOCK_SIZE_K
            w1_ptr += BLOCK_SIZE_K * n_cols
            w3_ptr += BLOCK_SIZE_K * n_cols
        return

    @torch.inference_mode()
    def kernel_ff(x: torch.Tensor, w1: torch.Tensor, rms_w: torch.Tensor, w3: torch.Tensor, o: torch.Tensor, scale: float, activation: str = "silu"):
        assert x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and x.is_contiguous()
        assert w1.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and w1.is_contiguous()
        assert w3.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and w3.is_contiguous()
        assert rms_w.dtype in [torch.float32, torch.float16, torch.bfloat16] and rms_w.is_contiguous()
        assert x.shape[-1] == w1.shape[1] and x.shape[-1] == w3.shape[0]
        assert rms_w.shape[-1] == w1.shape[0]
        n_cols = w1.shape[1]
        n_ctx = x.shape[-1]
        o = o.view(-1, n_cols)
        x = x.view(-1, n_cols)
        grid = lambda META: (triton.cdiv(n_ctx, META["BLOCK_SIZE_M"]) * triton.cdiv(n_cols, META["BLOCK_SIZE_N"]), )
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            ff_llama[grid](x, w1, rms_w, w3, o, scale, n_cols, n_ctx)
        else:
            with torch.cuda.device(x.device.index):
                from .triton_utils.kernels import convert_fp8_weights
                rms_w, w1, w3 = convert_fp8_weights(rms_w, w1, w3)
                ff_llama[grid](x, w1, rms_w, w3, o, scale, n_cols, n_ctx, IS_FP8=False)
        return
else:
    @triton.jit
    def ff_llama(x, w1, rms_w, w3, o, scale, n_cols: int, n_ctx: int, eps=1e-6, BLOCK_SIZE_M: tl.constexpr = 256, BLOCK_SIZE_N: tl.constexpr = 256, BLOCK_SIZE_K: tl.constexpr = 64, IS_FP8: tl.constexpr = False):
        # Kernel for computing F.silu(w1(x)) * w3(x) with RMS norm.
        range_m = tl.arange(0, BLOCK_SIZE_M)
        range_n = tl.arange(0, BLOCK_SIZE_N)
        rms_range = tl.arange(0, BLOCK_SIZE_K)
        m_mask = range_m[:, None] < n_ctx
        n_mask = range_n[None, :] < n_cols

        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        x_ptr = x + range_m * n_cols + range_n
        w1_ptr = w1 + range_n * n_cols + range_m
        rms_w_ptr = rms_w + rms_range
        w3_ptr = w3 + range_m * n_cols + range_n
        o_ptr = o + range_m * n_cols + range_n

        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE_K)):
            k_mask = (rms_range < n_cols) & (rms_range >= i * BLOCK_SIZE_K)
            rms_w_val = tl.load(rms_w_ptr + i * BLOCK_SIZE_K, mask=k_mask, other=1.0).to(tl.float32)
            x_val = tl.load(x_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            w1_val = tl.load(w1_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            w3_val = tl.load(w3_ptr, mask=m_mask & n_mask, other=0.0).to(tl.float32)

            x_val = rms_norm(x_val, rms_w_val)
            acc1 += tl.dot(x_val, w1_val, allow_tf32=False)
            acc2 += tl.dot(w1_val, x_val, allow_tf32=False)
            tl.store(o_ptr, (silu(acc1) * acc2).to(o_ptr.dtype.element_ty), mask=m_mask & n_mask)
            x_ptr += BLOCK_SIZE_K
            w1_ptr += BLOCK_SIZE_K * n_cols
            w3_ptr += BLOCK_SIZE_K * n_cols
        return

    @torch.inference_mode()
    def kernel_ff(x: torch.Tensor, w1: torch.Tensor, rms_w: torch.Tensor, w3: torch.Tensor, o: torch.Tensor, scale: float, activation: str = "silu"):
        assert x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and x.is_contiguous()
        assert w1.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and w1.is_contiguous()
        assert w3.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] and w3.is_contiguous()
        assert rms_w.dtype in [torch.float32, torch.float16, torch.bfloat16] and rms_w.is_contiguous()
        assert x.shape[-1] == w1.shape[1] and x.shape[-1] == w3.shape[0]
        assert rms_w.shape[-1] == w1.shape[0]
        n_cols = w1.shape[1]
        n_ctx = x.shape[-1]
        o = o.view(-1, n_cols)
        x = x.view(-1, n_cols)
        grid = lambda META: (triton.cdiv(n_ctx, META["BLOCK_SIZE_M"]) * triton.cdiv(n_cols, META["BLOCK_SIZE_N"]), )
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            ff_llama[grid](
