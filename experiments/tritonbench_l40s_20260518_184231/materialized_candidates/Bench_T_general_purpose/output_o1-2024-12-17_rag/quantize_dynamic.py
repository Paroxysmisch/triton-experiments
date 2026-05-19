import copy
import triton
import triton.language as tl
import torch


# ------------------------------------------------------------------------------
# Triton kernels from Document 1
# ------------------------------------------------------------------------------
def _get_autotune_configs():
    return [
        triton.Config({"num_warps": 1, "BLOCK_SIZE": 64}),
        triton.Config({"num_warps": 2, "BLOCK_SIZE": 128}),
        triton.Config({"num_warps": 4, "BLOCK_SIZE": 256}),
    ]

@triton.jit
def _abs_max(val1, val2):
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs

@triton.autotune(configs=_get_autotune_configs(), key=["M", "N"])
@triton.jit
def _triton_dynamic_quantize_kernel(
    output_ptr,  
    input_ptr,   
    scale_ptr,   
    stride_outputm,  
    stride_outputn,  
    stride_inputm,   
    stride_inputn,   
    n_elements,  
    M: tl.constexpr,  
    N: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, N)
    mask = offsets < n_elements
    input_ptrs = input_ptr + pid * stride_inputm + offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=1e-6)
    abs_max_f = tl.reduce(input_vals, 0, _abs_max)
    dynamic_per_token_scale = 127.0 / abs_max_f
    precison_mask = tl.where(input_vals > 0, 0.5, -0.5)
    output_vals = (input_vals * dynamic_per_token_scale + precison_mask).to(tl.int8)
    output_ptrs = output_ptr + pid * stride_outputm + offsets
    tl.store(output_ptrs, output_vals, mask=mask)
    tl.store(scale_ptr + pid, abs_max_f / 127.0)

def triton_dynamic_quantize(out, input, scale):
    assert input.is_contiguous(), "input must be contiguous"
    num_tokens = input.size(0)
    hidden_size = input.size(1)
    block_size = 1024
    if hidden_size & (hidden_size - 1) == 0 and hidden_size > 0:
        block_size = min(hidden_size // 2, block_size)
    else:
        hidden_size = 1 << (hidden_size - 1).bit_length()
        block_size = min(hidden_size // 2, block_size)
    _triton_dynamic_quantize_kernel[(num_tokens,)](
        out,
        input,
        scale,
        out.stride(0),
        out.stride(1),
        input.stride(0),
        input.stride(1),
        n_elements=input.size(1),
        M=num_tokens,
        N=hidden_size,
    )

# ------------------------------------------------------------------------------
# Wrapper function: quantize_dynamic
# ------------------------------------------------------------------------------
def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    """
    Converts a float model to a dynamic quantized model by replacing specified
    modules with their dynamic weight-only quantized versions.
    """

    if not inplace:
        model = copy.deepcopy(model)

    # If no mapping is provided, set a default mapping for common module types
    if mapping is None:
        mapping = {
            torch.nn.Linear: torch.nn.quantized.dynamic.Linear,
            torch.nn.LSTM: torch.nn.quantized.dynamic.LSTM,
            torch.nn.GRU: torch.nn.quantized.dynamic.GRU,
            torch.nn.RNN: torch.nn.quantized.dynamic.RNN,
        }

    def _should_quantize_submodule(name, module):
        # If qconfig_spec is None, default to quantizing known large-weight modules
        if qconfig_spec is None:
            return isinstance(module, tuple(mapping.keys()))
        # If qconfig_spec is a set, check module type/name
        if isinstance(qconfig_spec, set):
            if type(module) in qconfig_spec or name in qconfig_spec:
                return True
            return False
        # If qconfig_spec is a dict, check if module type or name is in keys
        if isinstance(qconfig_spec, dict):
            if type(module) in qconfig_spec or name in qconfig_spec:
                return True
            return False
        return False

    # Recursively apply dynamic quantization
    for name, submodule in model.named_children():
        if _should_quantize_submodule(name, submodule):
            submodule_type = type(submodule)
            q_type = mapping.get(submodule_type, None)
            if q_type is not None:
                # Replace with quantized version
                quantized_module = q_type.from_float(submodule)
                setattr(model, name, quantized_module)
        else:
            quantize_dynamic(submodule, qconfig_spec=qconfig_spec, inplace=True, mapping=mapping)

    return model
