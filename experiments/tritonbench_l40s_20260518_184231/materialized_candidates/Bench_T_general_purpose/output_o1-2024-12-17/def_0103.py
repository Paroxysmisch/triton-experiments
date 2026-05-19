import torch
import triton
import triton.language as tl

@triton.jit
def bitwise_and_binomial_kernel(
    input_ptr,
    other_ptr,
    probs_ptr,
    logits_ptr,
    output_ptr,
    input_other_broadcast_shape_0,
    input_other_broadcast_shape_1,
    probs_logits_broadcast_shape_0,
    probs_logits_broadcast_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Calculate the range of elements this program will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_other_broadcast_shape_0 * input_other_broadcast_shape_1
    
    # Broadcast indices for input and other
    input_idx = (offsets // input_other_broadcast_shape_1) % input_other_broadcast_shape_0
    other_idx = offsets % input_other_broadcast_shape_1
    input_val = tl.load(input_ptr + input_idx, mask=mask)
    other_val = tl.load(other_ptr + other_idx, mask=mask)
    bitwise_and = input_val & other_val
    
    # Broadcast indices for probs/logits
    probs_logits_idx = (offsets // probs_logits_broadcast_shape_1) % probs_logits_broadcast_shape_0
    probs_logits_idx = probs_logits_idx % probs_logits_broadcast_shape_1
    if probs_ptr != 0:
        p = tl.load(probs_ptr + probs_logits_idx, mask=mask)
    else:
        logits = tl.load(logits_ptr + probs_logits_idx, mask=mask)
        p = tl.sigmoid(logits)
    
    # Sample from Binomial distribution
    uniform = tl.rand(offsets, seed=123)
    binomial = tl.zeros_like(bitwise_and, dtype=tl.float32)
    for _ in range(tl.max(bitwise_and, 0) + 1):
        binomial += tl.where(uniform < p, 1.0, 0.0)
        uniform = tl.rand(offsets, seed=123 + _ + 1)
    
    tl.store(output_ptr + offsets, binomial, mask=mask)

def bitwise_and_binomial(
    input: torch.Tensor,
    other: torch.Tensor,
    probs: torch.Tensor = None,
    logits: torch.Tensor = None,
) -> torch.Tensor:
    assert probs is not None or logits is not None, "Either probs or logits must be provided"
    assert not (probs is not None and logits is not None), "Only one of probs or logits should be provided"
    
    # Compute bitwise_and and its broadcast shape
    bitwise_and = torch.bitwise_and(input, other)
    input_other_broadcast = torch.broadcast_shapes(input.shape, other.shape)
    probs_logits = probs if probs is not None else logits
    probs_logits_broadcast = torch.broadcast_shapes(bitwise_and.shape, probs_logits.shape)
    
    # Allocate output tensor
    output = torch.empty(probs_logits_broadcast, dtype=torch.float32, device=input.device)
    
    # Launch kernel
    def grid(meta):
        return (triton.cdiv(output.numel(), meta['BLOCK_SIZE']),)
    
    bitwise_and_binomial_kernel[grid](
        input,
        other,
        probs if probs is not None else 0,
        logits if logits is not None else 0,
        output,
        input_other_broadcast[0],
        input_other_broadcast[1],
        probs_logits_broadcast[0],
        probs_logits_broadcast[1],
        BLOCK_SIZE=1024,
    )
    return output
