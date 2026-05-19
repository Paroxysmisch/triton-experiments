import triton
import triton.language as tl
import triton.testing as tt

@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements,
                      BLOCK_SIZE, P2):
    row = tl.program_id(0)
    offsets = row * n_elements
    stride = grid(0).size * n_elements

    # Load elements of row from memory
    x = tl.load(x_ptr + offsets, mask=row < BLOCK_SIZE)

    # Compute max value and scale elements
    max_val = tl.max(x)
    tl.store(output_maxs + row, max_val)
    x = tl.extra.cuda.libdevice.llrint(x / max_val * 127)

    # Store elements back to memory
    tl.store(output_ptr + offsets, x, mask=row < BLOCK_SIZE)

def quantize_rowwise(x, BLOCK_SIZE=256):
    x_cu = tt.to_cuda(x)
    output = tt.to_cuda(np.zeros_like(x))
    output_maxs = tt.to_cuda(np.zeros((x.shape[0],)))
    P2 = int(np.ceil(np.log2(x.shape[1])))
    n_elements = x.shape[1]

    # Launch kernel
    _quantize_rowwise[((x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, )](
        x_cu, output, output_maxs, n_elements, BLOCK_SIZE, P2)

    return output, output_maxs
