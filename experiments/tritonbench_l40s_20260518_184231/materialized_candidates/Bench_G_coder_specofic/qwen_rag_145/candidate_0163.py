Here's the code as required:
        
    @triton.jit
    def leaky_relu(x):
        return tl.where(x >= 0, x, 0.01 * x)

    @triton.autotune(
        configs=get_autotune_config(),
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def matmul_kernel(
            a_ptr, b_ptr, c_ptr, M, N, K,
            stride_am, stride_ak,  stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,  ACTIVATION: tl.constexpr 
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def matmul(a, b, activation=""):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel[grid](
            a, b, c,  M, N, K,  a.stride(0), a.stride(1),  b.stride(0), b.stride(1),  c.stride(0), c.stride(1),
            ACTIVATION=activation 
        )
        return c

Note: These codes run on GPU using Triton and PyTorch. The matrix multiplication is performed in parallel on the GPU and then the results are accumulated into the resulting matrix. Trtions are used to create optimized CUDA kernels, varying in their block sizes for performance. Depending on the outputs, the `leaky_relu` function could be used as an activation for the output or not.

Also note that the tuning process in Triton could take some time, especially on large matrices. The chosen autotuned configurations are aimed to optimize performance, execution time, and the memory consumption, based on the input matrix dimensions, block sizes, and other conditions.

This code is under active development and may not be suitable for large-scale production use, but is suitable for research and development, especially for machine learning applications.
