import triton
import triton.language as tl

@triton.jit
def scale_weights_kernel(weights_ptr, scale_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Position within the grid
    pid = tl.program_id(0)
    
    # Offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    weights = tl.load(weights_ptr + offsets, mask=offsets < n_elements, other=0.0)
    scale = tl.load(scale_ptr)
    
    # Scale weights
    scaled_weights = weights * scale
    
    # Store results
    tl.store(output_ptr + offsets, scaled_weights, mask=offsets < n_elements)

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    """
    Converts a float model to a dynamic quantized model by replacing specified modules with their dynamic weight-only quantized versions.
    
    Args:
        model: input model
        qconfig_spec: Either a dictionary mapping submodule names/types to quantization configurations or a set of types/names for dynamic quantization
        inplace: carry out model transformations in-place, mutating the original module
        mapping: maps submodule types to dynamically quantized versions
        
    Returns:
        Model: The quantized model.
    """
    # Import necessary libraries for model manipulation
    import copy
    from torch.quantization import quantize_dynamic as torch_quantize_dynamic
    
    # Ensure model is not modified in-place unless specified
    if not inplace:
        model = copy.deepcopy(model)
    
    # Perform dynamic quantization
    quantized_model = torch_quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=None,  # dtype is overridden by qconfig_spec if provided
        mapping=mapping
    )
    
    return quantized_model
