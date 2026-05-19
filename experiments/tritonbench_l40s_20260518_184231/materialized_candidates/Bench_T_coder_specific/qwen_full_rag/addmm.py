import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(
    alpha,
    beta,
    IS_BETA_ZERO: tl.constexpr,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    tiles_per_cta: tl.constexpr,
    VALUES_STRIDE_0: tl.constexpr,
    VALUES_STRIDE_1: tl.constexpr,
    CROW_INDEXES_STRIDE: tl.constexpr,
    COL_INDEXES_STRIDE: tl.constexpr,
    values_ptr,
    crow_indices_ptr,
    col_indices_ptr,
    mat1_ptr,
    mat2_ptr,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    # Get program ids
    pid_ctas = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get row block id and start index of nnz elements corresponding to the row blocks computed by this CTA
    row_block_id = pid_ctas * tiles_per_cta + pid_row
    row_start_nnz_idx = tl.load(crow_indices_ptr + row_block_id * CROW_INDEXES_STRIDE)
    row_end_nnz_idx = tl.load(crow_indices_ptr + (row_block_id + 1) * CROW_INDEXES_STRIDE)

    # Compute the row indices of the blocks processed by this CTA
    mid_row_idx = row_block_id * BLOCKSIZE_ROW + tl.arange(0, BLOCKSIZE_ROW)

    # For each block, there is a mini-loop unrolled over COLUMN_BLOCK_SIZE tiles
    for k in range(0, K, BLOCKSIZE_COL):

        # Compute the column indices of the blocks processed by this thread block
        mid_col_idx = k + tl.arange(0, BLOCKSIZE_COL)

        # Load values of the block matrix C = α*A @ B + β*C
        # Note that the active elements of each block are determined by the stride of the values tensor.
        values_block_idx = (
            row_start_nnz_idx
            + mid_row_idx[:, None] * VALUES_STRIDE_0
            + mid_col_idx[None, :] * VALUES_STRIDE_1
        )
        a_block_ptr = values_block_idx
        b_block_ptr = values_block_idx

        # Load values of the matrix A
        a_values = tl.load(
            mat1_ptr
            + mid_row_idx[:, None] * mat1_ptr.stride(0)
            + (k + tl.arange(0, BLOCKSIZE_COL))[None, :] * mat1_ptr.stride(1),
            mask=(mid_row_idx[:, None] < M) & (mid_col_idx[None, :] < K),
            other=0.0,
        )

        # Load values of the matrix B
        b_values = tl.load(
            mat2_ptr
            + (k + tl.arange(0, BLOCKSIZE_COL)) * mat2_ptr.stride(0)
            + mid_col_idx[None, :] * mat2_ptr.stride(1),
            mask=(mid_row_idx[:, None] < K) & (mid_col_idx[None, :] < N),
            other=0.0,
        )

        # Accumulate the result of the current block into the accumulator
        acc_block = tl.dot(a_values, b_values, allow_tf32=allow_tf32, out_dtype=acc_dtype)
        if IS_BETA_ZERO:
            acc_block *= alpha
        else:
            c_values = tl.load(values_ptr + values_block_idx)
            acc_block = alpha * acc_block + beta * c_values

        # Store the result
        tl.store(values_ptr + values_block_idx, acc_block)


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    f_name = "addmm"

    def check(cond, msg):
        if not cond:
            raise ValueError(msg)

    def check_device(f_name, t, device):
        check(
            t.device == device and t.device.type == "cuda",
            f"{f_name}(): all inputs are expected to be on the same CUDA device.",
        )

    def check_mm_compatible_shapes(f_name, lhs, rhs):
        check(
            lhs.dim() >= 2 and rhs.dim() >= 2,
            f"{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, "
            f"but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}."
        )

        m, k = lhs.shape[-2:]
        n = rhs.shape[-1]

        check(
            k == rhs.shape[-2],
            f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, "
            f"got lhs.shape[-1] == {k} which is not equal to rhs.shape[-2] == {rhs.shape[-2]}.",
        )

    def check_dtype(f_name, t, dtype, *additional_dtypes):
        check(
            t.dtype == dtype
            and t.dtype in ((torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes)),
            f"{f_name}(): all inputs are expected to be of the same dtype "
            f"and one of (half, bfloat16, float32) or {additional_dtypes}, "
            f"but got dtype == {t.dtype}.",
        )

    def make_contiguous(t):
        if not t.is_contiguous():
            return t.contiguous()
        else:
            return t

    def broadcast_batch_dims(f_name, *tensors):
        try:
            return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
        except Exception:
            check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")

    def prepare_inputs(*dense_tensors):
        # Introduce fake batch dimension if not present for convenience.
        tensors = [make_contiguous(t.unsqueeze(0)) for t in dense_tensors]

        # Compute broadcasted batch dimension
        batch_dims_broadcasted = broadcast_batch_dims(f_name, *tensors)

        # Broadcast batch dimensions and squash.
        # The result can be either a view or a copy.
        def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
            return t.broadcast_to(batch_dims + invariant_dims).flatten(
                0, len(batch_dims) - 1
            )

        tensors = [
            batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:]) for t in tensors
        ]

        return tensors

    def broadcast_batch_dims_dense(f_name, sparsity_meta, *dense_tensors):
        batch_shape = broadcast_batch_dims(f_name, *dense_tensors)

        size = batch_shape + sparsity_meta["size"]
        crow_indices = torch.zeros(size=(sparsity_meta["num_blocks"] + 1,), dtype=torch.int32, device=dense_tensors[0].device)
        col_indices = torch.zeros(size=sparsity_meta["nnz"], dtype=torch.int32, device=dense_tensors[0].device)
        values = torch.zeros(size=sparsity_meta["value_size"], dtype=dense_tensors[0].dtype, device=dense_tensors[0].device)
        sparsity_meta_out = {
            "crow_indices": crow_indices,
            "col_indices": col_indices,
            "values": values,
            "size": sparsity_meta["size"],
            "num_blocks": sparsity_meta["num_blocks"],
            "nnz": sparsity_meta["nnz"],
            "blocksize": sparsity_meta["blocksize"],
        }
        return sparsity_meta_out, prepare_inputs(*dense_tensors)

    check_device(f_name, input, mat1.device)
    check_device(f_name, mat1, mat2.device)

    if not skip_checks:
        check_dtype(f_name, input, mat1.dtype, mat2.dtype)
        check_mm_compatible_shapes(f_name, mat1, mat2)
        if out is not None:
            check_device(f_name, out, mat1.device)
            check_dtype(f_name, out, input.dtype)
            check(
                out.shape == input.shape
                and out._nnz() == input._nnz(),
                f"{f_name}(): Expects `out` to be of shape {input.shape} "
                f"and with nnz equal to {input._nnz()} "
                f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}"
            )

    if out is None:
        out = input
    elif out is not input:
        out.copy_(input)

    if out.numel() == 0 or out._nnz() == 0:
        return out

    sparsity_meta_out, mat1, mat2 = broadcast_batch_dims_dense(f_name, out.sparsity_meta(), mat1, mat2)

    blocksize = out.values().shape[-2:]
    m = mat1.size(-2)
    n = mat2.size(-1)
    k = mat1.size(-1())

    # NOTE: (m, 0) @ (0, n) == zeros(m, n)
    if alpha == 0.0 or k == 0:
        out.values().mul_(
