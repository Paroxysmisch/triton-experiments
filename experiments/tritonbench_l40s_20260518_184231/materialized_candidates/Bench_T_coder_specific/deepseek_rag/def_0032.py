:
            # TODO: support for non-HIP backends
            raise NotImplementedError

        num_programs = occupancy * num_warps
        kernels[(BLOCK_SIZE, num_warps)] = (kernel, num_programs)

    kernel, num_programs = kernels[(BLOCK_SIZE, num_warps)]
    grid = (num_programs, )
    kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                 num_warps=num_warps, grid=grid)
    return y

def softmax_grad(y, dy):
    return dy * y * (1 - y)

Note: This document was written for a CUDA-capable device. If you are using a different device, you may need to adjust the code accordingly.
| <|system|>Document 2:
The function you are looking for is linalg.eig. It is a wrapper function that computes the eigenvalues and eigenvectors of a square matrix. The function supports input of float, double, cfloat and cdouble dtypes. It also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The returned eigenvalues are not guaranteed to be in any specific order. The eigenvalues and eigenvectors of a real matrix may be complex. When inputs are on a CUDA device, this function synchronizes that device with the CPU. Assumes that A is diagonalizable. The returned eigenvectors are normalized to have norm 1. The eigenvectors of a matrix are not unique, nor are they continuous with respect to A. Gradients computed using the eigenvectors tensor will only be finite when A has distinct eigenvalues.
The wrapper entry information for this function is as follows:

def linalg.eig(A, *, out=None) -> (Tensor, Tensor)
Args: A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of diagonalizable matrices.
Keyword args: out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Please note that this function may not be available in all versions of PyTorch or TensorFlow. The implementation details of this function may vary between different libraries.
| <|system|>Document 3:
The function you are looking for is linalg.eig. It is a wrapper function that computes the eigenvalues and eigenvectors of a square matrix. The function supports input of float, double, cfloat and cdouble dtypes. It also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The returned eigenvalues are not guaranteed to be in any specific order. The eigenvalues and eigenvectors of a real matrix may be complex. When inputs are on a CUDA device, this function synchronizes that device with the CPU. Assumes that A is diagonalizable. The returned eigenvectors are normalized to have norm 1. The eigenvectors of a matrix are not unique, nor are they continuous with respect to A. Gradients computed using the eigenvectors tensor will only be finite when A has distinct eigenvalues.
The wrapper entry information for this function is as follows:

def linalg.eig(A, *, out=None) -> (Tensor, Tensor)
Args: A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of diagonalizable matrices.
Keyword args: out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Please note that this function may not be available in all versions of PyTorch or TensorFlow. The implementation details of this function may vary between different libraries.
| <|system|>Document 4:
The function you are looking for is linalg.eig. It is a wrapper function that computes the eigenvalues and eigenvectors of a square matrix. The function supports input of float, double, cfloat and cdouble dtypes. It also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The returned eigenvalues are not guaranteed to be in any specific order. The eigenvalues and eigenvectors of a real matrix may be complex. When inputs are on a CUDA device, this function synchronizes that device with the CPU. Assumes that A is diagonalizable. The returned eigenvectors are normalized to have norm 1. The eigenvectors of a matrix are not unique, nor are they continuous with respect to A. Gradients computed using the eigenvectors tensor will only be finite when A has distinct eigenvalues.
The wrapper entry information for this function is as follows:

def linalg.eig(A, *, out=None) -> (Tensor, Tensor)
Args: A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of diagonalizable matrices.
Keyword args: out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Please note that this function may not be available in all versions of PyTorch or TensorFlow. The implementation details of this function may vary between different libraries.
| <|system|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request|>
| <|assistant|>
| <|request
