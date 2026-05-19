import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    
    x = tl.load(input_ptr + block_start, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + e^{-x})
    x = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + block_start, x, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    """
    sigmoid_argmax(input, dim=None, keepdim=False) -> LongTensor
    Applies the sigmoid function elementwise on 'input', then computes the
    indices of the maximum values along 'dim'. If 'dim' is None, finds the 
    index of the maximum value in the flattened tensor.
    """
    # Ensure input is a CUDA tensor
    assert input.is_cuda, "Input must be a CUDA tensor."

    # Flatten the tensor only if we need to do a global argmax
    if dim is None:
        # Create output buffer for sigmoid results
        flat_inp = input.contiguous().view(-1)
        n_elements = flat_inp.numel()
        out = torch.empty_like(flat_inp, device=flat_inp.device)

        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _sigmoid_kernel[grid](
            flat_inp,
            out, 
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE
        )
        # Return global argmax index
        return torch.argmax(out).long()

    else:
        # Apply sigmoid along entire tensor first
        inp_contig = input.contiguous()
        n_elements = inp_contig.numel()
        out = torch.empty_like(inp_contig, device=inp_contig.device)

        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _sigmoid_kernel[grid](
            inp_contig,
            out,
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE
        )
        out = out.view_as(inp_contig)
        # Now compute argmax along the given dimension
        return torch.argmax(out, dim=dim, keepdim=keepdim).long()
