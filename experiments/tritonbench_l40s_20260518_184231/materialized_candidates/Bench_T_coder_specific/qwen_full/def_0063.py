import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List
from flag_gems.utils.shape_utils import broadcast_shapes

def generate_wrapper(func_inputs: dict) -> Tuple[str, str]:
    wrapper, kernel = generate_wrapper_and_kernel(func_inputs)
    assert wrapper_inputs_match_kernel(func_inputs, kernel)
    return wrapper, kernel

def generate_wrapper_and_kernel(func_inputs: dict) -> Tuple[str, str]:
    a, b, dims = func_inputs['args']
    rank_a, rank_b = a.ndim, b.ndim
    if type(dims) is int:
        num_dims = dims
        dim_a = list(range(rank_a - num_dims, rank_a))
        dim_b = list(range(0, num_dims))
    elif type(dims[0]) is int:
        dim_a, dim_b = dims
    else:
        dim_a, dim_b = [list(d) for d in dims]
    assert all([i < rank_a for i in dim_a])
    assert all([i < rank_b for i in dim_b])
    a_is_contiguous = all([i in a.stride() for i in sorted(dim_a)])
    b_is_contiguous = all([i in b.stride() for i in sorted(dim_b)])

    shape_a, shape_b = broadcast_shapes(a.shape, b.shape)
    shape_out = list(shape_a)
    for i, j in zip(dim_a, dim_b):
        shape_out[i] = shape_b[j]
    a_expanded = expand_tensor(a, shape_a)
    b_expanded = expand_tensor(b, shape_b)
    a_strides, b_strides = a_expanded.stride(), b_expanded.stride()
    wrapper = textwrap.dedent(f"""
        import torch
        import triton
        import triton.language as tl
        from flag_gems.utils.shape_utils import broadcast_shapes

        @triton.jit
        def kernel(a_ptr, b_ptr, output_ptr, {', '.join(f'shape_{i}' for i in range(len(shape_out)))},
                    {', '.join(f'stride_a_{i}' for i in range(len(dim_a)))},
                    {', '.join(f'stride_b_{i}' for i in range(len(dim_b)))},
                    {', '.join(f'stride_out_{i}' for i in range(len(shape_out)))},
                    BLOCK_SIZE: tl.constexpr, 
                    NUM_SM: tl.constexpr):
            pid = tl.program_id(0)
            num_pid = tl.num_programs(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < {shape_out[0]}
            {('            offset = tl.where(mask, offset, 0)\n' if len(shape_out) == 1 else '')}
            {'            '.join(f'index_out_{i} = offset // {shape_out[i+1] if len(shape_out) > 1 else 1}' for i in range(len(shape_out) - 1))}
            index_out_0 = offset
            {('            '.join(f'offset %= {shape_out[i+1] if len(shape_out) > 1 else 1}' for i in range(len(shape_out) - 1)))}
            {'            '.join(f'a_offset = index_out_{i} * stride_a_{i}' for i in range(len(dim_a)))}
            {'            '.join(f'b_offset = index_out_{i} * stride_b_{i}' for i in range(len(dim_b)))}
            {'            '.join(f'output_offset = index_out_{i} * stride_out_{i}' for i in range(len(shape_out)))}
            a_ptr += a_offset
            b_ptr += b_offset
            output_ptr += output_offset
            a_tile = tl.load(a_ptr + tl.arange(0, {max(1, triton.next_power_of_2(len(dim_a)))}), mask=a_offset < {shape_a[0]}, other=0)
            b_tile = tl.load(b_ptr + tl.arange(0, {max(1, triton.next_power_of_2(len(dim_b)))}), mask=b_offset < {shape_b[0]}, other=0)
            output = tl.dot(a_tile, b_tile)
            tl.store(output_ptr, output, mask=mask)

        def call_kernel(a, b):
            shape_a, shape_b = broadcast_shapes(a.shape, b.shape)
            {dims_code}
            a_expanded = {expand_code}
            b_expanded = {expand_code}
            output = torch.empty({shape_out}, device=a.device, dtype=a.dtype)
            grid = lambda meta: (triton.cdiv({shape_out[0]}, meta['BLOCK_SIZE']),)
            dtype = a.dtype
            if dtype == torch.float16:
                BLOCK_SIZE = 128
                NUM_SM = 84
            elif dtype == torch.bfloat16:
                BLOCK_SIZE = 128
                NUM_SM = 84
            elif dtype == torch.float32:
                BLOCK_SIZE = 64
                NUM_SM = 168
            kernel[grid](a_expanded, b_expanded, output,
                         {', '.join(f'shape_{i}' for i in range(len(shape_out)))},
                         {', '.join(f'stride_a_{i}' for i in range(len(dim_a)))},
                         {', '.join(f'stride_b_{i}' for i in range(len(dim_b)))},
                         {', '.join(f'stride_out_{i}' for i in range(len(shape_out)))},
                         BLOCK_SIZE=BLOCK_SIZE, NUM_SM=NUM_SM)
            return output
        """)
    kernel = textwrap.dedent(f"""
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(a_ptr, b_ptr, output_ptr, {', '.join(f'shape_{i}' for i in range(len(shape_out)))},
                    {', '.join(f'stride_a_{i}' for i in range(len(dim_a)))},
                    {', '.join(f'stride_b_{i}' for i in range(len(dim_b)))},
                    {', '.join(f'stride_out_{i}' for i in range(len(shape_out)))},
                    BLOCK_SIZE: tl.constexpr, 
                    NUM_SM: tl.constexpr):
            pid = tl.program_id(0)
            num_pid = tl.num_programs(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < {shape_out[0]}
            {('            offset = tl.where(mask, offset, 0)\n' if len(shape_out) == 1 else '')}
            {'            '.join(f'index_out_{i} = offset // {shape_out[i+1] if len(shape_out) > 1 else 1}' for i in range(len(shape_out) - 1))}
            index_out_0 = offset
            {('            '.join(f'offset %= {shape_out[i+1] if len(shape_out) > 1 else 1}' for i in range(len(shape_out) - 1)))}
            {'            '.join(f'a_offset = index_out_{i} * stride_a_{i}' for i in range(len(dim_a)))}
            {'            '.join(f'b_offset = index_out_{i} * stride_b_{i}' for i in range(len(dim_b)))}
            {'            '.join(f'output_offset = index_out_{i} * stride_out_{i}' for i in range(len(shape_out)))}
            a_ptr += a_offset
            b_ptr += b_offset
            output_ptr += output_offset
            a_tile = tl.load(a_ptr + tl.arange(0, {max(1, triton.next_power_of_2(len(dim_a)))}), mask=a_offset < {shape_a[0]}, other=0)
            b_tile = tl.load(b_ptr + tl.arange(0, {max(1, triton.next_power_of_2(len(dim_b)))}), mask=b_offset < {shape_b[0]}, other=0)
            output = tl.dot(a_tile, b_tile)
            tl.store(output_ptr, output, mask=mask)
        """)
    return wrapper, kernel

def wrapper_inputs_match_kernel(func_inputs: dict, kernel: str) -> bool:
    a, b, dims = func_inputs['args']
    rank_a, rank_b = a.ndim, b.ndim
    if type(dims) is int:
        num_dims = dims
        dim_a = list(range(rank_a - num_dims, rank_a))
        dim_b = list(range(0, num_dims))
    elif type(dims[0]) is int:
        dim_a, dim_b = dims
    else:
        dim_a, dim_b = [list(d) for d in dims]
    num_pid = 'num_pid'
    if rank_a == 0:
        num_pid = '1'
    elif len(dim_a) > 0:
        size_last_dim_a = [f'shape_a[{i}]' for i in dim_a if i != dim_a[-1]]
        size_last_dim_a += [f'shape_a[{dim_a[-1]}] // {func_inputs["device_properties"]["multiprocessor_count"]}']
        num_pid = ' * '.join(size_last_dim_a)
    num_pid = f'tl.cdiv({shape_out[0]}, BLOCK_SIZE)'
    return num_pid in kernel
