import logging
import triton
import triton.language as tl

_mode_dim_mapping = {'reduced': [0, 1], 'complete': [1, 0], 'r': []}

def qr(A, mode='reduced', *, out=None):
    A = _extend_internal(A)
    if out is not None:
        assert len(out) == 2, "out should be a tuple of two tensors"
        assert all(map(lambda x: x.ndim >= 2, out)), "out should be tensors of at least order 2"
        assert A.shape[:-2] == out[0].shape[:-2] and A.shape[:-2] == out[1].shape[:-2], "inconsistent shape"
        assert A.shape[-1] == out[0].shape[-1] and A.shape[-2] == out[1].shape[-2], "inconsistent shape"
        assert A.is_contiguous() and out[0].is_contiguous() and out[1].is_contiguous(), "one of the output tensors is not contiguous"
    else:
        out = (empty_like(A), triton.empty(A.shape[:-2] + (min(A.shape[-2:]),), dtype=A.dtype))
    assert A.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128], "A must be float or complex type"
    assert mode in _mode_dim_mapping, "mode must be 'reduced', 'complete', or 'r'"
    if mode == 'r':
        assert A.shape[-2] >= A.shape[-1], "for mode 'r', m (A.shape[-2]) must be >= n (A.shape[-1])"
    _qr_batched(A, mode, out[0], out[1])
    return out if out else (out[0].squeeze(list(range(out[0].ndim)[-2:]))), (out[1].squeeze(list(range(out[1].ndim)[-2:])))
