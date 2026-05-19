import torch
import triton
import triton.language as tl
from einops import rearrange
from packaging import version

if version.parse(triton.__version__) >= version.parse("2.1.0"):

    @triton.jit
    def _bmm_chunk_bwd_kernel(
        a_ptr, dout_ptr, db_ptr, res_ptr, stride_a_batch, stride_a_csize, stride_a_h, stride_a_m, stride_a_n, stride_dout_b, stride_dout_csize_m, stride_dout_csize_n, stride_dout_h, stride_dout_m, stride_dout_n, stride_db_batch, stride_db_csize, stride_db_h, stride_db_n, stride_db_m, stride_res_batch, stride_res_csize, stride_res_h, stride_res_m, stride_res_n, batch, csize, h, m, n, HAS_RESIDUAL: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
    ):
        pid_b = tl.program_id(axis=1)
        pid_c = tl.program_id(axis=2)
        pid_h = tl.program_id(axis=3)
        pid_m = tl.program_id(axis=4)
        pid_n = tl.program_id(axis=5)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_cs = pid_c * BLOCK_SIZE_CS + tl.arange(0, BLOCK_SIZE_CS)
        a_offs = offs_cs[:, None, None] * stride_a_csize + offs_m[None, :, None] * stride_a_m + offs_n[None, None, :] * stride_a_n
        dout_offs = offs_cs[:, None, None] * stride_dout_csize_m + offs_m[None, :, None] * stride_dout_m + offs_n[None, None, :] * stride_dout_n
        db_offs = offs_cs[:, None, None] * stride_db_csize + offs_m[None, :, None] * stride_db_m + offs_n[None, None, :] * stride_db_n + offs_m[:, None, None] * stride_db_h + offs_n[None, None, :] * stride_db_n + pid_b * stride_db_batch
        a_ptrs = a_ptr + a_offs
        dout_ptrs = dout_ptr + dout_offs
        db_ptrs = db_ptr + db_offs
        if HAS_RESIDUAL:
            res_offs = offs_cs[:, None, None] * stride_res_csize + offs_m[None, :, None] * stride_res_m + offs_n[None, None, :] * stride_res_n + offs_m[:, None, None] * stride_res_h + offs_n[None, None, :] * stride_res_n + pid_b * stride_res_batch
            res_ptrs = res_ptr + res_offs
        acc = tl.zeros((BLOCK_SIZE_CS, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for h_ in range(0, h, 2):
            a = tl.load(a_ptrs + h_ * stride_a_h)
            dout = tl.load(dout_ptrs + h_ * stride_dout_h)
            acc += tl.dot(a, dout)
        if HAS_RESIDUAL:
            res = tl.load(res_ptrs)
            acc += res
        tl.store(db_ptrs, acc.to(db_ptrs.dtype.element_ty))

    def _bmm_chunk_bwd(a, dout, db, residual=None):
        B, C, H, M, N = a.shape
        assert a.shape == dout.shape
        assert a.shape == db.shape
        if residual is not None:
            assert residual.shape == a.shape
            assert residual.stride(-1) == 1
        assert dout.stride(-1) == 1
        assert db.stride(-1) == 1
        a = a.contiguous()
        dout = dout.contiguous()
        db = db.contiguous()
        if residual is not None:
            residual = residual.contiguous()
        grid = lambda META: (B, triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]), H, M, N)
        BLOCK_SIZE_CS = min(triton.next_power_of_2(C), 16)
        with torch.cuda.device(a.device.index):
            _bmm_chunk_bwd_kernel[grid](
                a,
                dout,
                db,
                residual,
                stride_a_batch=a.stride(0),
                stride_a_csize=a.stride(1),
                stride_a_h=a.stride(2),
                stride_a_m=a.stride(3),
                stride_a_n=a.stride(4),
                stride_dout_b=dout.stride(0),
                stride_dout_csize_m=dout.stride(1),
                stride_dout_csize_n=dout.stride(2),
                stride_dout_h=dout.stride(3),
                stride_dout_m=dout.stride(4),
                stride_dout_n=dout.stride(5),
                stride_db_batch=db.stride(0),
                stride_db_csize=db.stride(1),
                stride_db_h=db.stride(3),
                stride_db_m=db.stride(4),
                stride_db_n=db.stride(5),
                stride_res_batch=residual.stride(0) if residual is not None else 0,
                stride_res_csize=residual.stride(1) if residual is not None else 0,
                stride_res_h=residual.stride(2) if residual is not None else 0,
                stride_res_m=residual.stride(3) if residual is not None else 0,
                stride_res_n=residual.stride(4) if residual is not None else 0,
                batch=B,
                csize=C,
                h=H,
                m=M,
                n=N,
                HAS_RESIDUAL=residual is not None,
                BLOCK_SIZE_CS=BLOCK_SIZE_CS,
            )
