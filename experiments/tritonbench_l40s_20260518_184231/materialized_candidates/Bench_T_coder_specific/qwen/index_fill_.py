import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=4),
    ],
    key=['N', 'M', 'K', 'dim']
)
def index_fill(
    x_ptr: tl.Pointer,
    index_ptr: tl.Pointer,
    value: float32,
    N: tl.int32,
    M: tl.int32,
    K: tl.int32,
    dim: tl.int32,
    stride_x: tl.Tensor[3, tl.int32],
    stride_index: tl.Tensor[2, tl.int32]
):
    index_fill_kernel[x_ptr, (N * M * K)](
        x_ptr,
        stride_x,
        index_ptr,
        stride_index,
        value,
        N,
        M,
        K,
        dim
    )

def index_fill_(self, dim, index, value):
    # Convert PyTorch tensors to Triton tensors
    x_ptr = self.data.ptr
    index_ptr = index.data.ptr
    stride_x = self.stride()
    stride_index = index.stride()

    # Call the Triton kernel
    index_fill(x_ptr, index_ptr, value, self.shape[0], self.shape[1], self.shape[2], dim, stride_x, stride_index)
