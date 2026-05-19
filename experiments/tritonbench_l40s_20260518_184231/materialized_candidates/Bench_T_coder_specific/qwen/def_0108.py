import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def affine_grid(theta_ptr, grid_ptr, N, H_out, W_out):
    pid = tl.program_id(0)
    row = pid // W_out
    col = pid % W_out
    batch = row // H_out
    y = row % H_out
    
    # Extract affine parameters
    theta = tl.load(theta_ptr + batch * 6)
    
    # Compute grid coordinates
    x = (col - 0.5) / W_out * 2.0 - 1.0
    y = (y - 0.5) / H_out * 2.0 - 1.0
    
    # Apply affine transformation
    u = theta[0] * x + theta[1] * y + theta[2]
    v = theta[3] * x + theta[4] * y + theta[5]
    
    # Normalize coordinates to be within [0, 1]
    u = (u + 1.0) / 2.0
    v = (v + 1.0) / 2.0
    
    # Store grid coordinates
    tl.store(grid_ptr + pid * 2, u)
    tl.store(grid_ptr + pid * 2 + 1, v)

@triton.jit
def grid_sample_with_affine(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out, mode='bilinear'):
    pid = tl.program_id(0)
    n = pid // (C * H_out * W_out)
    c = (pid // (H_out * W_out)) % C
    h = (pid // W_out) % H_out
    w = pid % W_out
    
    u = tl.load(grid_ptr + (h * W_out + w) * 2)
    v = tl.load(grid_ptr + (h * W_out + w) * 2 + 1)
    
    u *= W_in - 1
    v *= H_in - 1
    
    u_floored = tl.floor(u).astype(tl.int32)
    v_floored = tl.floor(v).astype(tl.int32)
    u_ceiled = u_floored + 1
    v_ceiled = v_floored + 1
    
    u_floored = tl.clip(u_floored, 0, W_in - 1)
    v_floored = tl.clip(v_floored, 0, H_in - 1)
    u_ceiled = tl.clip(u_ceiled, 0, W_in - 1)
    v_ceiled = tl.clip(v_ceiled, 0, H_in - 1)
    
    i0 = n * C * H_in * W_in + c * H_in * W_in + v_floored * W_in + u_floored
    i1 = n * C * H_in * W_in + c * H_in * W_in + v_floored * W_in + u_ceiled
    i2 = n * C * H_in * W_in + c * H_in * W_in + v_ceiled * W_in + u_floored
    i3 = n * C * H_in * W_in + c * H_in * W_in + v_ceiled * W_in + u_ceiled
    
    val0 = tl.load(input_ptr + i0)
    val1 = tl.load(input_ptr + i1)
    val2 = tl.load(input_ptr + i2)
    val3 = tl.load(input_ptr + i3)
    
    u_frac = u - u_floored
    v_frac = v - v_floored
    
    if mode == 'bilinear':
        val = (1 - u_frac) * (1 - v_frac) * val0 + \
              u_frac * (1 - v_frac) * val1 + \
              (1 - u_frac) * v_frac * val2 + \
              u_frac * v_frac * val3
    elif mode == 'nearest':
        idx = tl.select(u_frac < 0.5, u_floored, u_ceiled)
        idy = tl.select(v_frac < 0.5, v_floored, v_ceiled)
        val = tl.load(input_ptr + n * C * H_in * W_in + c * H_in * W_in + idy * W_in +
