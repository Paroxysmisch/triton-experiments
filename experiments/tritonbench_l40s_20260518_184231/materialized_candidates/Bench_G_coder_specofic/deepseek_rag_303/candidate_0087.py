import torch
import triton
import triton.language as tl

TritonKLDivergence = None  # Variable to hold the compiled triton kernel function
TRITON_AVAILABLE = False

try:
    import triton
    from triton import language as tl

    class _TritonKLDivergence(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, log_target, weight, reduction):
            input = input.contiguous()
            target = target.contiguous()
            log_target = log_target.contiguous()
            kwargs = {'input': input, 'log_targs': log_target, 'reduction': reduction}
            if weight is not None:
                weight = weight.contiguous()
                kwargs['weight'] = weight

            NUM_CTAS = 1
            NUM_WARPS = 4
            BLOCK_SIZE = 256
            output = torch.empty_like(input)
            forward_triton(input, **kwargs, output=output, num_ctas=NUM_CTAS, num_warps=NUM_WARPS, block_size=BLOCK_SIZE)
            ctx.save_for_backward(input, target, log_target, weight)
            ctx.reduction = reduction
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (input, target, log_target, weight) = ctx.saved_tensors
            if grad_output.numel() == 1:
                return backward_triton(input, grad_output, target, log_target, weight, str(ctx.reduction))
            else:
                return backward_triton(input, grad_output, target, log_target, weight, str(ctx.reduction)) * grad_output

    class _reduction:
        NONE = "none"
        SUM = "sum"
        MEAN = "mean"
        BATCHMEAN = "batchmean"

    class _TritonKLDivergenceCached:
        cache = {}  # Cache loaded functions by key.

        @staticmethod
        def get(key):
            if key not in _TritonKLDivergenceCached.cache:
                _TritonKLDivergenceCached.cache[key] = _TritonKLDivergence.apply
            return _TritonKLDivergenceCached.cache[key]

    _str_to_reduction_mode = {
        "none": _reduction.NONE,
        "sum": _reduction.SUM,
        "mean": _reduction.MEAN,
        "batchmean": _reduction.BATCHMEAN,
    }

    def celu(x):  # Custom activation: Exponential linear unit (CELU)
        return torch.nn.functional.relu(x) + torch.nn.functional.relu(-torch.exp(x)) + 1

    class celu_act(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            out = celu(A)
            ctx.save_for_backward(A)
            return out

        @staticmethod
        def backward(ctx, out_grad):
            (in_A,) = ctx.saved_tensors
            sgn = torch.sign(in_A)
            mask1 = (in_A > 0).to(torch.float32)
            mask2 = (in_A <= 0).to(dtype=torch.float32)
            in_A_exp = torch.exp(in_A)
            in_A_celu = in_A + (in_A_exp - 1) * mask1
            in_B = sgn * celu(in_A_exp)
            return out_grad * in_B

    def celu_wrapper(A):
        return celu_act.apply(A)

    TRITON_AVAILABLE = True
except Exception as e:
    print(f"Couldn't load TRITON, likely because your CUDA is too old. {repr(e)}")

 Truncated(import torch
import triton
import triton.language as tl

TritonKLDivergence = None  # Variable to hold the compiled triton kernel function
TRITON_AVAILABLE = False

try:
    import triton
    from triton import language as tl

    class _TritonKLDivergence(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, log_target, weight, reduction):
            input = input.contiguous()
            target = target.contiguous()
            log_target = log_target.contiguous()
            kwargs = {'input': input, 'log_targs': log_target, 'reduction': reduction}
            if weight is not None:
                weight = weight.contiguous()
                kwargs['weight'] = weight

            NUM_CTAS = 1
            NUM_WARPS = 4
            BLOCK_SIZE = 256
            output = torch.empty_like(input)
            forward_triton(input, **kwargs, output=output, num_ctas=NUM_CTAS, num_warps=NUM_WARPS, block_size=BLOCK_SIZE)
            ctx.save_for_backward(input, target, log_target, weight)
            ctx.reduction = reduction
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (input, target, log_target, weight) = ctx.saved_tensors
            if grad_output.numel() == 1:
                return backward_triton(input, grad_output, target, log_target, weight, str(ctx.reduction))
            else:
                return backward_triton(input, grad_output, target, log_target, weight, str(ctx.reduction)) * grad_output

    class _reduction:
        NONE = "none"
        SUM = "sum"
        MEAN = "mean"
        BATCHMEAN = "batchmean"

    class _TritonKLDivergenceCached:
        cache = {}  # Cache loaded functions by key.

        @staticmethod
        def get(key):
            if key not in _TritonKLDivergenceCached.cache:
                _TritonKLDivergenceCached.cache[key] = _TritonKLDivergence.apply
            return _TritonKLDivergenceCached.cache[key]

    _str_to_reduction_mode = {
        "none": _reduction.NONE,
        "sum": _reduction.SUM,
        "mean": _reduction.MEAN,
        "batchmean": _reduction.BATCHMEAN,
    }

    def celu(x):  # Custom activation: Exponential linear unit (CELU)
        return torch.nn.functional.relu(x) + torch.nn.functional.relu(-torch.exp(x)) + 1

    class celu_act(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            out = celu(A)
            ctx.save_for_backward(A)
            return out

        @staticmethod
        def backward(ctx, out_grad):
            (in_A,) = ctx.saved_tensors
            sgn = torch.sign(in_A)
            mask1 = (in_A > 0).to(torch.float32)
            mask2 = (in_A <= 0).to(dtype=torch.float32)
            in_A_exp = torch.exp(in_A)
            in_A_celu = in_A + (in_A_exp - 1) * mask1
            in_B = sgn * celu(in_A_exp)
            return out_grad * in_B

    def celu_wrapper(A):
        return celu_act.apply(A)

    TRITON_AVAILABLE = True
except Exception as e:
    print(f"Couldn't load TRITON, likely because your CUDA is too old. {repr(
