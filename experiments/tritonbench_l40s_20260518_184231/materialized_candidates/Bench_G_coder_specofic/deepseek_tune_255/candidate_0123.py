import torch
import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def chunk_global_reversed_cumsum_vector_kernel(
        s,
        z,
        B: tl.constexpr,
        H: tl.constexpr,
        T: tl.constexpr,
        S: tl.constexpr,
        BT: tl.constexpr,
        BS: tl.constexpr,
        dtype: tl.constexpr,
    ):
        i_bh = tl.program_id(0)
        i_bh % B
        i_bh // B
        i_s = tl.program_id(1)
        i_s * BT
        i_t = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        o_i = i_b * B * H * T + i_h * T + i_t * BT + tl.arange(0, BT)

        p_s = (
            s + o_i[:, None, None] * S
            + (i_b * H + i_h)[None, :, None] * T * S
            + (i_s)[None, None, :] * BS
        )
        p_z = (
            z + o_i[:, None, None] * S
            + (i_b * H + i_h)[None, :, None] * T * S
            + (i_s)[None, None, :] * BS
        )
        b_s = tl.load(p_s, mask=(o_i[:, None, None] < T * S), other=0.0).to(
            tl.float32
        )
        m_s = tl.tril(tl.ones((BT, BS), dtype=tl.float32))
        b_s = tl.dot(b_s, m_s)
        tl.store(p_z, b_s.to(dtype), mask=(o_i[:, None, None] < T * S))

        b_z = tl.sum(b_s, axis=1)
        for i in range(BT):
            p_z = (
                z
                + (i_b * H + i_h) * T * S
                + i_t * BT * S
                + i_s * BS
                + (i_t * BT + i) * S
                + tl.arange(0, BS)[:, None]
            )
            tl.store(
                p_z,
                b_z.to(dtype),
                mask=(i_t * BT + i) < T
                * S
                & (i_s * BS + tl.arange(0, BS)[None, :]) < S,
            )

    def chunk_global_reversed_cumsum_vector(s, dtype=None):
        B, H, T, S = s.shape
        BT = 32
        BS = 32
        z = torch.empty((B, H, T, S), device=s.device, dtype=dtype)
        chunk_global_reversed_cumsum_vector_kernel[(B * H, S, triton.cdiv(T, BT))](
            s,
            z,
            B,
            H,
            T,
            S,
            BT,
            BS,
            dtype=dtype if dtype else s.dtype,
        )
        return z
