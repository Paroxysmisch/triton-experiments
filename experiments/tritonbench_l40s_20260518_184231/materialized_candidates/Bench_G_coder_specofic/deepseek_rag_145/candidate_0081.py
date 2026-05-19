@triton.jit
def layer_norm_bwd_dx_fused(
    grad_output_ptr, 
    input_ptr, 
    weight_ptr, 
    mean_ptr, 
    rstd_ptr, 
    grad_input_ptr, 
    num_elements, 
    eps,
    block_size,
    ):

    # determine thread index and size in the grid
    pid = tl.program_id(0)
    grid_size = tl.grid_size(0)

    # calculates the number of blocks in the grid
    num_blocks = num_elements // block_size

    # computation start index
    start = pid * block_size
 
    # load mean and rstd
    means = tl.load(mean_ptr + start)
    rstds = tl.load(rstd_ptr + start)

    for i in range(start, num_elements, grid_size):
        input = tl.load(input_ptr + i)
        weight = tl.load(weight_ptr + i)
        grad_output = tl.load(grad_output_ptr + i)
        grad_input = (grad_output * weight - (input - means) * rstds * tl.sum((input - means) * grad_output)) / (rstds * rstds + eps)
        tl.store(grad_input_ptr + i, grad_input)
