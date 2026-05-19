import triton
import triton.language as tl
import torch

@triton.jit
def fftn(input, s=None, dim=None, norm=None, *, out=None):
    pass

def wrapper_fftnd(input, s=None, dim=None, norm=None):
    if input.dtype == torch.half or input.dtype == torch.cfloat:
        assert input.device.type == "cuda"
        arch = input.device.get_device_properties().major
        assert (
            arch >= 7
        ), "cuFFT only supports computing FFT on GPU with architecture 7.0 or above."
        assert input.is_contiguous()
        if dim is None:
            dim = list(range(input.ndim))
        if s is None:
            s = [dim_size for dim_size, dim in zip(input.shape, dim)]
        assert all(
            [s_i > 0 and isinstance(s_i, int) for s_i in s]
        ), "Signal size should be a non-negative integer"
        valid_last_dims = [dim % input.ndim for dim in dim]
        assert set(valid_last_dims) == set(
            dim
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -input.ndim, input.ndim - 1, dim
        )
        complex_dtype = torch.complex64 if input.dtype == torch.half else torch.complex128
        if out is None:
            out = torch.empty_like(input, dtype=complex_dtype)
        elif out.dtype != complex_dtype:
            raise RuntimeError("Expected output dtype to be {}, but got {}".format(complex_dtype, out.dtype))
        assert out.is_contiguous()

        # Prepare arguments for cuFFT
        handle = libtorch_triton_frontend.get_current_stream()
        plan = cuFFTPlan(input, s=s, dims=dim, norm=norm, stream=handle)
        batch_strides = [input.stride(i) for i in range(len(dim))]
        in_base_ptr = input
        out_base_ptr = out
        n_elements = libtorch_triton_utils volume(s)
        run_cuFFT(plan, in_base_ptr, out_base_ptr, batch_strides, n_elements, handle)
        return out
    else:
        return torch.fft.fftn(input, s, dim, norm, *, out=out)
