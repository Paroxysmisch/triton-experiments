import triton as tt

@tt.kernel
def rand_kernel(Z):
    pid = tt.program_id(0)
    n = Z.shape[0]
    for i in range(pid, n, tt.num_programs()):
        Z[i] = tt.rand()

def rand(*size, **kwargs):
    Z = tt.zeros(size, **kwargs)
    rand_kernel[Z.shape[0], 128](Z)
    return Z
