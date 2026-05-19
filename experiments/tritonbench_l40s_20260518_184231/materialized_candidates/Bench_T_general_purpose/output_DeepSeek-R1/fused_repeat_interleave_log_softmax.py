import torch
import torch.nn.functional as F

def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    # Handle dim=None by flattening the input tensor
    if dim is None:
        input = input.flatten()
        dim = 0
    
    # Ensure repeats is a tensor on the same device as input
    repeats = torch.as_tensor(repeats, device=input.device, dtype=torch.long)
    
    # Compute the max along the specified dimension, keep dimensions for broadcasting
    max_val = torch.max(input, dim=dim, keepdim=True).values
    
    # Shift input by max for numerical stability
    shifted = input - max_val
    
    # Compute exp(shifted) and multiply by repeats, then sum along the dimension
    exp_shifted = torch.exp(shifted)
    # Expand repeats to match the shape of exp_shifted for broadcasting
    repeat_expander = [1] * input.ndim
    repeat_expander[dim] = input.shape[dim]
    repeats_expanded = repeats.view(repeat_expander).expand_as(exp_shifted)
    
    sum_exp = (exp_shifted * repeats_expanded).sum(dim=dim, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    
    # Compute log-softmax values for the original elements
    log_softmax_vals = shifted - log_sum_exp
    
    # Repeat interleave the log-softmax values along the specified dimension
    output = torch.repeat_interleave(log_softmax_vals, repeats, dim=dim)
    
    # Handle output tensor if provided
    if out is not None:
        if not out.is_contiguous():
            out.copy_(output)
        else:
            out = output
        return out
    
    return output

# Example usage:
# input = torch.randn(2, 3, requires_grad=True)
# repeats = torch.tensor([2, 1, 3])
# output = fused_repeat_interleave_log_softmax(input, repeats, dim=1)
# print(output)
# output.sum().backward()
# print(input.grad)
