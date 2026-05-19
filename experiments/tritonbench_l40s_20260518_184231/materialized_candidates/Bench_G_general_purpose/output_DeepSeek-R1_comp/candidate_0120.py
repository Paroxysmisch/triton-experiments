import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,
    z_ptr,
    B,
    H,
    T,
    stride_s_b,
    stride_s_h,
    stride_s_t,
    stride_z_b,
    stride_z_h,
    stride_z_t,
    BT: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    off_b = b_idx
    off_h = h_idx

    running_sum = tl.zeros(tl.float32, 1)  # Use float32 for accumulation to prevent overflow

    num_chunks = tl.cdiv(T, BT)
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * BT
        end_t = tl.minimum(start_t + BT, T)
        chunk_size = end_t - start_t

        # Create block pointers for the current chunk
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, H, T),
            strides=(stride_s_b, stride_s_h, stride_s_t),
            offsets=(off_b, off_h, start_t),
            block_shape=(1, 1, chunk_size),
            order=(0, 1, 2),
        )
        z_block_ptr = tl.make_block_ptr(
            base=z_ptr,
            shape=(B, H, T),
            strides=(stride_z_b, stride_z_h, stride_z_t),
            offsets=(off_b, off_h, start_t),
            block_shape=(1, 1, chunk_size),
            order=(0, 1, 2),
        )

        # Load the current chunk
        chunk = tl.load(s_block_ptr, boundary_check=(2,), padding_option="zero")

        # Compute local cumulative sum
        local_cumsum = tl.cumsum(chunk, axis=2)

        # Add the running sum to the local cumulative sum
        global_cumsum = local_cumsum + running_sum

        # Store the result
        tl.store(z_block_ptr, global_cumsum.to(z_ptr.dtype.element_ty), boundary_check=(2,))

        # Update the running sum with the sum of the current chunk
        sum_val = tl.sum(chunk)
        running_sum += sum_val

def chunk_global_cumsum_scalar(s: torch.Tensor, dtype: torch.dtype = None, BT: int = 1024) -> torch.Tensor:
    if dtype is None:
        dtype = s.dtype
    B, H, T = s.shape
    z = torch.empty_like(s, dtype=dtype)

    grid = (B, H)
    chunk_global_cumsum_scalar_kernel[grid](
        s, z,
        B, H, T,
        s.stride(0), s.stride(1), s.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        BT=BT,
    )
    return z
