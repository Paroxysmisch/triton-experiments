import triton
import triton.language as tl

@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    q_start_loc_ptr, q_seq_length_ptr, kv_seq_length_ptr, block_offsets_ptr,
    k_scales_zeros_ptr, v_scales_zeros_ptr,
    batch_size, max_num_blocks, num_heads, head_dim, block_size,
    q_seq_length, kv_seq_length, quant_policy,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // max_num_blocks
    block_id = pid % max_num_blocks

    if bid >= batch_size:
        return

    q_start_loc = tl.load(q_start_loc_ptr + bid)
    q_seq_length = tl.load(q_seq_length_ptr + bid)
    kv_seq_length = tl.load(kv_seq_length_ptr + bid)
    block_offset = tl.load(block_offsets_ptr + block_id)

    k_cache_ptr = k_caches_ptr + bid * num_heads * head_dim * kv_seq_length + block_offset * head_dim
    v_cache_ptr = v_caches_ptr + bid * num_heads * head_dim * kv_seq_length + block_offset * head_dim

    for head in range(num_heads):
        k_state_ptr = k_states_ptr + bid * num_heads * head_dim * q_seq_length + head * head_dim * q_seq_length
        v_state_ptr = v_states_ptr + bid * num_heads * head_dim * q_seq_length + head * head_dim * q_seq_length

        for i in range(block_size):
            if q_start_loc + block_offset * block_size + i < q_seq_length:
                k_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = k_state_ptr[(q_start_loc + block_offset * block_size + i) * head_dim + tl.arange(0, head_dim)]
                v_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = v_state_ptr[(q_start_loc + block_offset * block_size + i) * head_dim + tl.arange(0, head_dim)]

@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    q_start_loc_ptr, q_seq_length_ptr, kv_seq_length_ptr, block_offsets_ptr,
    k_scales_zeros_ptr, v_scales_zeros_ptr,
    batch_size, max_num_blocks, num_heads, head_dim, block_size,
    q_seq_length, kv_seq_length, quant_policy,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // max_num_blocks
    block_id = pid % max_num_blocks

    if bid >= batch_size:
        return

    q_start_loc = tl.load(q_start_loc_ptr + bid)
    q_seq_length = tl.load(q_seq_length_ptr + bid)
    kv_seq_length = tl.load(kv_seq_length_ptr + bid)
    block_offset = tl.load(block_offsets_ptr + block_id)

    k_cache_ptr = k_caches_ptr + bid * num_heads * head_dim * kv_seq_length + block_offset * head_dim
    v_cache_ptr = v_caches_ptr + bid * num_heads * head_dim * kv_seq_length + block_offset * head_dim

    k_scale_ptr = k_scales_zeros_ptr + bid * num_heads * 2
    v_scale_ptr = v_scales_zeros_ptr + bid * num_heads * 2

    for head in range(num_heads):
        k_state_ptr = k_states_ptr + bid * num_heads * head_dim * q_seq_length + head * head_dim * q_seq_length
        v_state_ptr = v_states_ptr + bid * num_heads * head_dim * q_seq_length + head * head_dim * q_seq_length

        k_scale = tl.load(k_scale_ptr + head * 2)
        k_zero = tl.load(k_scale_ptr + head * 2 + 1)
        v_scale = tl.load(v_scale_ptr + head * 2)
        v_zero = tl.load(v_scale_ptr + head * 2 + 1)

        for i in range(block_size):
            if q_start_loc + block_offset * block_size + i < q_seq_length:
                k_state = k_state_ptr[(q_start_loc + block_offset * block_size + i) * head_dim + tl.arange(0, head_dim)]
                v_state = v_state_ptr[(q_start_loc + block_offset * block_size + i) * head_dim + tl.arange(0, head_dim)]

                if quant_policy == 4:
                    k_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = (k_state * k_scale + k_zero).to(tl.int4)
                    v_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = (v_state * v_scale + v_zero).to(tl.int4)
                elif quant_policy == 8:
                    k_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = (k_state * k_scale + k_zero).to(tl.int8)
                    v_cache_ptr[head_dim * i + tl.arange(0, head_dim)] = (v_state * v_scale + v_zero).to(tl.int8)

### Python Wrapper


import torch

# Example tensors
batch_size = 2
num_heads = 4
head_dim = 64
q_seq_length = 128
kv_seq_length = 256
max_num_blocks = 4
block_size = 32

k_states = torch.randn(batch_size, num_heads, head_dim, q_seq_length, device='cuda')
v_states = torch.randn(batch_size, num_heads, head_dim, q_seq_length, device='cuda')
k_caches = torch.zeros(batch_size, num_heads, head_dim, kv_seq_length, device='cuda')
v_caches = torch.zeros(batch_size, num_heads, head_dim, kv_seq_length, device='cuda')
q_start_loc = torch.tensor([0, 0], device='cuda')
q_seq_length = torch.tensor([q_seq_length, q_seq_length], device='cuda')
kv_seq_length = torch.tensor([kv_seq_length, kv_seq_length], device='cuda')
block_offsets = torch.tensor([0, 32, 64, 96], device='cuda')

# Optional quantization parameters
k_scales_zeros = torch.randn(batch_size, num_heads, 2, device='cuda')
v_scales_zeros = torch.randn(batch_size, num_heads, 2, device='cuda')

# Fill the cache
k_caches, v_caches = fill_kv_cache(
    k_states, v_states, k_caches, v_caches,
    q_start_loc, q_seq_length, kv_seq_length, block_offsets,
    k_scales_zeros, v_scales_zeros, quant_policy=4
)

print(k_caches)
print(v_caches)
