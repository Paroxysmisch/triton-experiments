if use_fp8:
            a = a * a_scale[None, :]
            b = b * b_scale[None, :]
        accumulator += tl.dot(a, b, transpose_a=True, transpose_b=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptr_local = tl.load(c_ptr + offs_token_id[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_ptr_local += accumulator * MUL_ROUTED_WEIGHT
    tl.store(c_ptr + offs_token_id[:, None] * stride_cm + offs_bn[None, :] * stride_cn, c_ptr_local)

def invoke_fused_moe_kernel(
    a,
    b,
    c,
    a_scale,
    b_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    MUL_ROUTED_WEIGHT,
    top_k,
    compute_type,
    use_fp8,
):
    a_ptr = tl.pointer(a, tl.float32)
    b_ptr = tl.pointer(b, tl.float32)
    c_ptr = tl.pointer(c, tl.float32)
    a_scale_ptr = tl.pointer(a_scale, tl.float32)
    b_scale_ptr = tl.pointer(b_scale, tl.float32)
    topk_weights_ptr = tl.pointer(topk_weights, tl.float32)
    sorted_token_ids_ptr = tl.pointer(sorted_token_ids, tl.int32)
    expert_ids_ptr = tl.pointer(expert_ids, tl.int32)
    num_tokens_post_padded_ptr = tl.pointer(num_tokens_post_padded, tl.int32)
    grid = launch_grid(EM, BLOCK_SIZE_M)
    fused_moe_kernel[grid](
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N,
        K,
        EM,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT,
        top_k,
        compute_type,
        use_fp8,
    )
<|endoftext|>

TrionUser: I am an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition. The function first decomposes A into L and D through LDL decomposition, reconstructs matrix A, and then uses `torch.linalg.solve` to solve the linear system.
Wrapper Entry Information: solve_symmetric_ldl(A, b, *, hermitian=False, out=None) -> Tensor A (Tensor): 形状为 (*, n, n) 的对称（或 Hermitian）矩阵，其中 * 是零个或多个批次维度。 b (Tensor): 形状为 (*, n) 或 (*, n, k) 的右端项张量。 hermitian (bool, 可选): 是否将 A 视为 Hermitian 矩阵。默认值：False。 out (Tensor, 可选): 输出张量。如果为 None，则忽略。默认值：None。
Math: Given a symmetric (or Hermitian) matrix A in \mathbb{K}^{n \times n} (where \mathbb{K} is the real field \mathbb{R} or complex field \mathbb{C}), the LDL decomposition of A is represented as: A = L D L^{\mathrm{T}} or A = L D L^{\mathrm{H}}.
other: This function supports batch processing; all computations are performed across batch dimensions.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 2:
TrionUser: I am an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition. The function first decomposes A into L and D through LDL decomposition, reconstructs matrix A, and then uses `torch.linalg.solve` to solve the linear system.
Wrapper Entry Information: solve_symmetric_ldl(A, b, *, hermitian=False, out=None) -> Tensor A (Tensor): 形状为 (*, n, n) 的对称（或 Hermitian）矩阵，其中 * 是零个或多个批次维度。 b (Tensor): 形状为 (*, n) 或 (*, n, k) 的右端项张量。 hermitian (bool, 可选): 是否将 A 视为 Hermitian 矩阵。默认值：False。 out (Tensor, 可选): 输出张量。如果为 None，则忽略。默认值：None。
Math: Given a symmetric (or Hermitian) matrix A in \mathbb{K}^{n \times n} (where \mathbb{K} is the real field \mathbb{R} or complex field \mathbb{C}), the LDL decomposition of A is represented as: A = L D L^{\mathrm{T}} or A = L D L^{\mathrm{H}}.
other: This function supports batch processing; all computations are performed across batch dimensions.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 3:
TrionUser: I am an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition. The function first decomposes A into L and D through LDL decomposition, reconstructs matrix A, and then uses `torch.linalg.solve` to solve the linear system.
Wrapper Entry Information: solve_symmetric_ldl(A, b, *, hermitian=False, out=None) -> Tensor A (Tensor): 形状为 (*, n, n) 的对称（或 Hermitian）矩阵，其中 * 是零个或多个批次维度。 b (Tensor): 形状为 (*, n) 或 (*, n, k) 的右端项张量。 hermitian (bool, 可选): 是否将 A 视为 Hermitian 矩阵。默认值：False。 out (Tensor, 可选): 输出张量。如果为 None，则忽略。默认值：None。
Math: Given a symmetric (or Hermitian) matrix A in \mathbb{K}^{n \times n} (where \
