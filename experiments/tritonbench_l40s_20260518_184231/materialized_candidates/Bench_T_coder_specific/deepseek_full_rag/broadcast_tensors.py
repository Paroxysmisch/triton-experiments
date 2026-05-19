)

    def check_dtype_supported(f_name, input, t):
        check(
            t in [torch.bfloat16, torch.float16, torch.float32],
            f"{f_name}(): input.dtype should be one of the following: bfloat16, float16, float32.",
        )

    def check_dim_compatible(f_name, input, t):
        check(
            input.dim() == t.dim(),
            f"{f_name}(): input and sparse must have the same number of dimensions: "
            f"expected at least dimension {t.dim()} but got "
            f"dimension {input.dim()} instead",
        )

    def check_size_compatible(f_name, input, t):
        check(
            input.size(0) == t.size(0),
            f"{f_name}(): Expects sparse size ({t.size(0)}) to match input size ({input.size(0)}).",
        )

    def check_device_match(f_name, input, t):
        check(
            input.device == t.device,
            f"{f_name}(): Expects device of input and sparse to be the same.",
        )

    def check_mat1_2_dim(f_name, mat1, mat2):
        check(
            mat1.dim() == 3 and mat2.dim() == 3,
            f"{f_name}(): mat1 and mat2 must be 3D tensors.",
        )

    def check_mat1_2_bf16_fp16(f_name, mat1, mat2):
        check(
            all(x in [torch.bfloat16, torch.float16] for x in [mat1.dtype, mat2.dtype]),
            f"{f_name}(): mat1 and mat2 must be of dtype bfloat16 or float16.",
        )

    def check_mat1_2_size(f_name, mat1, mat2):
        check(
            mat1.size(2) == mat2.size(1),
            f"{f_name}(): Expects size(-1) of mat1 ({mat1.size(-1)}) to match size(1) of mat2 ({mat2.size(1)}).",
        )

    def check_input_dim(f_name, input):
        check(
            input.dim() >= 3,
            f"{f_name}(): Expects input to be at least a 3D tensor.",
        )

    def check_input_device(f_name, input):
        check(
            input.is_cpu() or input.is_cuda(),
            f"{f_name}(): Expects input to be on CPU or CUDA.",
        )

    def check_input_dtype(f_name, input):
        check(
            input.dtype in [torch.bfloat16, torch.float16, torch.float32],
            f"{f_name}(): input.dtype should be one of the following: bfloat16, float16, float32.",
        )

    def check_beta_zero(f_name, beta):
        check(
            beta == 0.0,
            f"{f_name}(): beta must be 0.0.",
        )

    if not skip_checks:
        check_bsr_layout(f_name, input)
        check_dtype_supported(f_name, input, mat1)
        check_dtype_supported(f_name, input, mat2)
        check_dim_compatible(f_name, input, mat1)
        check_dim_compatible(f_name, input, mat2)
        check_size_compatible(f_name, input, mat1)
        check_size_compatible(f_name, input, mat2)
        check_device_match(f_name, input, mat1)
        check_device_match(f_name, input, mat2)
        check_mat1_2_dim(f_name, mat1, mat2)
        check_mat1_2_bf16_fp16(f_name, mat1, mat2)
        check_mat1_2_size(f_name, mat1, mat2)
        check_input_dim(f_name, input)
        check_input_device(f_name, input)
        check_input_dtype(f_name, input)
        check_beta_zero(f_name, beta)

    k = mat2.size(-1)

    if out is None:
        out = torch.empty_like(input)
    else:
        check_input_dim(f_name, out)
        check_input_device(f_name, out)
        check_input_dtype(f_name, out)
        check(
            out.size() == input.size(),
            f"{f_name}(): out and input must have the same size.",
        )

    if max_grid is None:
        max_grid = (None, None, None)

    if input.is_cuda():
        tiled_shapes = (1, 1, 1, 1, k)
    else:
        tiled_shapes = (1, 1, 1, 1, 1)

    grid = lambda META: (
        triton.cdiv(input.size(1), META["BLOCKSIZE_ROW"] * tiled_shapes[1]),
        triton.cdiv(input.size(2), META["BLOCKSIZE_COL"] * tiled_shapes[4]),
        input.size(0),
    )

    _sampled_addmm_kernel[grid](
        alpha,
        beta,
        beta == 0.0,
        input.size(2),
        input.size(3),
        k,
        triton.next_power_of_2(k),
        input,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.indices().data_ptr(),
        input.indices().stride(0),
        input.indices().stride(1),
        input.indices().indices().data_ptr(),
        input.indices().indices().stride(0),
        input.indices().indices().stride(1),
        mat1,
        mat1.stride(0),
        mat1.stride(1),
        mat1.stride(2),
        mat1.stride(3),
        mat2,
        mat2.stride(0),
        mat2.stride(1),
        mat2.stride(2),
        mat2.stride(3),
        acc_dtype=torch.float32,
        allow_tf32=mat1.dtype in [torch.bfloat16, torch.float16],
        max_grid=max_grid,
    )

    return out
