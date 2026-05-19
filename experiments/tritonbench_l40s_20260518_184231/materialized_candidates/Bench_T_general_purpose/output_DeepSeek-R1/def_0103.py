import torch
import triton
import triton.language as tl

@triton.jit
def bitwise_and_binomial_kernel(
    input_ptr, other_ptr, output_ptr,
    total_count, probs_ptr, logits_ptr,
    input_other_broadcast_shape_x, input_other_broadcast_shape_y,
    probs_logits_broadcast_shape_x, probs_logits_broadcast_shape_y,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    
    # Calculate the range of elements this program will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (input_other_broadcast_shape_x * input_other_broadcast_shape_y)
    
    # Load input and other, compute bitwise AND
    input_offset = offsets // input_other_broadcast_shape_y % input_other_broadcast_shape_x
    other_offset = offsets % input_other_broadcast_shape_y
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0)
    other_val = tl.load(other_ptr + other_offset, mask=mask, other=0)
    bitwise_and_result = input_val & other_val
    
    # Load probs or logits based on availability
    if probs_ptr is not None:
        probs_offset = offsets // probs_logits_broadcast_shape_y % probs_logits_broadcast_shape_x
        prob = tl.load(probs_ptr + probs_offset, mask=mask, other=0)
    else:
        logits_offset = offsets // probs_logits_broadcast_shape_y % probs_logits_broadcast_shape_x
        logit = tl.load(logits_ptr + logits_offset, mask=mask, other=0)
        prob = tl.sigmoid(logit)
    
    # Generate binomial sample using inverse transform
    uniform = tl.rand(offsets, n_offsets=BLOCK_SIZE)
    binomial_sample = tl.floor(uniform * (total_count + 1))  # Simplified approximation
    
    # Combine with bitwise_and result (assuming multiplicative interaction)
    output = bitwise_and_result * binomial_sample
    tl.store(output_ptr + offsets, output, mask=mask)

def bitwise_and_binomial(
    input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor,
    probs: torch.Tensor = None, logits: torch.Tensor = None
) -> torch.Tensor:
    # Compute broadcast shapes for input/other and probs/logits
    input_other_broadcast = torch.broadcast_shapes(input.shape, other.shape)
    probs_logits = probs if probs is not None else logits
    if probs_logits is None:
        raise ValueError("Either probs or logits must be provided")
    probs_logits_broadcast = torch.broadcast_shapes(input_other_broadcast, probs_logits.shape)
    
    # Allocate output tensor
    output = torch.empty(probs_logits_broadcast, dtype=torch.long, device=input.device)
    
    # Flatten tensors for kernel processing
    flat_input = input.broadcast_to(input_other_broadcast).contiguous()
    flat_other = other.broadcast_to(input_other_broadcast).contiguous()
    flat_probs = probs.broadcast_to(probs_logits_broadcast).contiguous() if probs is not None else None
    flat_logits = logits.broadcast_to(probs_logits_broadcast).contiguous() if logits is not None else None
    
    # Launch kernel
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    bitwise_and_binomial_kernel[grid](
        flat_input, flat_other, output,
        total_count.item(),
        flat_probs if probs is not None else None,
        flat_logits if logits is not None else None,
        input_other_broadcast[0], input_other_broadcast[1],
        probs_logits_broadcast[0], probs_logits_broadcast[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
