// ... existing code ...

# In the forward kernel, added quantization after normalization:
    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    # Added quantization logic
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
    y = tl.math.round(y * scale)  # Quantize to INT8
    y = tl.maximum(tl.minimum(y, 127), -128) / scale  # Clamp and dequantize

// ... existing code ...

# In the backward kernel, handle quantization during output recomputation:
    if RECOMPUTE_OUTPUT:
        y = xhat * w if HAS_WEIGHT else xhat
        if HAS_BIAS:
            y = y + b
            
        # Added quantization logic in backward pass
        scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
        y = tl.math.round(y * scale)
        y = tl.maximum(tl.minimum(y, 127), -128) / scale

// ... existing code ...
