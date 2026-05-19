@triton.jit
def max_kernel(x_ptr, y_ptr, z_ptr, N, stride_x, stride_y, stride_z, offset_x, offset_y, offset_z, BLOCK_SIZE):
    pid = tl.program_id()
    n = tl.program_id()
    x = tl.load(x_ptr + pid * stride_x + offset_x)
    y = tl.load(y_ptr + pid * stride_y + offset_y)
    z = tl.load(z_ptr + pid * stride_z + offset_z)
    if x > y:
        if x > z:
            tl.store(y_ptr + pid * stride_y + offset_y, x)
            tl.store(z_ptr + pid * stride_z + offset_z, pid)
        else:
            tl.store(y_ptr + pid * stride_y + offset_y, z)
            tl.store(z_ptr + pid * stride_z + offset_z, pid)
    else:
        if y > z:
            tl.store(y_ptr + pid * stride_y + offset_y, y)
            tl.store(z_ptr + pid * stride_z + offset_z, pid)
        else:
            tl.store(y_ptr + pid * stride_y + offset_y, z)
            tl.store(z_ptr + pid * stride_z + offset_z, pid)

def max(input, dim, keepdim=False):
    values = input.clone()
    indices = input.new_zeros(input.size())
    stride = input.stride(dim)
    for i in range(input.size(dim)):
        max_kernel[(input.size(dim) - 1) // BLOCK_SIZE + 1](input.data_ptr() + i * stride, values.data_ptr(), indices.data_ptr(), input.size(dim), stride, values.stride(dim), indices.stride(dim), i * stride, i * values.stride(dim), i * indices.stride(dim), BLOCK_SIZE)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices
