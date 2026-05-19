import torch
import triton
import triton.language as tl
from triton.ops.fused_embedding_add_tanh import fused_embedding_add_tanh

@triton.jit
def kernel(other, output,  #
           EMBEDDING_TABLE_SEGMENT_LENGTH,  #
           BLOCK_SIZE_M,  #
           BLOCK_SIZE_N,  #
           stride_zm, stride_zn,  #
           stride_em, stride_en,  #
           stride_om, stride_on,  #
           MASK_M,  #
           MASK_N,  #
           *,  #
           IS_SPARSE: tl.constexpr,  #
           BLOCK_SIZE_SPARSE_M: tl.constexpr,  #
           BLOCK_SIZE_SPARSE_N: tl.constexpr  #
           ):
    # Map program ids to the block of indices to compute.
    pid = tl.program_id(0)
    mid = tl.program_id(1)
    nid = tl.program_id(2)
    # Compute the number of indices to compute in this program.
    num_indices = tl.cdiv(EMBEDDING_TABLE_SEGMENT_LENGTH, BLOCK_SIZE_M)
    # Compute the block offsets.
    m_offsets = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Compute the index bounds for this program.
    m_bounds = mid * BLOCK_SIZE_M, min(
        (mid + 1) * BLOCK_SIZE_M, EMBEDDING_TABLE_SEGMENT_LENGTH)
    # The write-back mask for this program.
    write_mask = (m_offsets[:, None] >= m_bounds[0]) & (
        m_offsets[:, None] < m_bounds[1])
    # Load the input data; we perform these loads in work-group row-major order.
    other = tl.load(other + m_offsets[:, None] * stride_om +
                    n_offsets[None, :] * stride_on,
                    mask=(m_offsets[:, None] < EMBEDDING_TABLE_SEGMENT_LENGTH) & (
                        n_offsets[None, :] < EMBEDDING_TABLE_SEGMENT_LENGTH),
                    other=0.0)
    # Compute the intermediate result.
    intermediate = tl.sum(other, axis=1)[:, None]
    # Write-back the result.
    output = tl.load(output + m_offsets[:, None] * stride_zm +
                     n_offsets[None, :] * stride_zn,
                     mask=write_mask)
    # Write-back the result.
    tl.store(output + m_offsets[:, None] * stride_zm +
             n_offsets[None, :] * stride_zn, output, mask=write_mask)

def call(indices,  #
         weight,  #
         other,  #
         *,  #
         padding_idx=None,  #
         max_norm=None,  #
         norm_type=2.0,  #
         scale_grad_by_freq=False,  #
         sparse=False,  #
         out=None  #
         ):
    output = torch.empty_like(indices, device='cuda')
    kernel[(indices.numel(), )](  #
        other, output,  #
        EMBEDDING_TABLE_SEGMENT_LENGTH=weight.size(0),  #
        BLOCK_SIZE_M=128,  #
        BLOCK_SIZE_N=128,  #
        stride_zm=output.stride(0), stride_zn=output.stride(1),  #
        stride_em=other.stride(0), stride_en=other.stride(1),  #
        MASK_M=(indices.reshape(-1) != padding_idx).to(torch.int32),  #
        MASK_N=(indices.reshape(-1) != padding_idx).to(torch.int32),  #
        IS_SPARSE=sparse,  #
        BLOCK_SIZE_SPARSE_M=32,  #
        BLOCK_SIZE_SPARSE_N=32,  #
    )
    return output
