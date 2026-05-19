import torch
import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(theta_ptr, grid_ptr, N, C, Hin, Win, Hout, Wout, align_corners: tl.constexpr):
    pid = tl.program_id(0)
    if pid < N * Hout * Wout:
        n = pid // (Hout * Wout)
        y = (pid % (Hout * Wout)) // Wout
        x = (pid % (Hout * Wout)) % Wout

        theta_offset = n * 6
        theta = tl.load(theta_ptr + theta_offset + tl.arange(0, 6)).to(tl.float32)

        if align_corners:
            x_norm = 2.0 * (x / (Wout - 1.0)) - 1.0
            y_norm = 2.0 * (y / (Hout - 1.0)) - 1.0
        else:
            x_norm = 2.0 * (x + 0.5) / Wout - 1.0
            y_norm = 2.0 * (y + 0.5) / Hout - 1.0
        
        grid_x = theta[0] * x_norm + theta[1] * y_norm + theta[2]
        grid_y = theta[3] * x_norm + theta[4] * y_norm + theta[5]

        grid_offset = pid * 2
        tl.store(grid_ptr + grid_offset, grid_x)
        tl.store(grid_ptr + grid_offset + 1, grid_y)


def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    """
    Applies an affine transformation to the input tensor followed by grid sampling.
    """
    N, C, Hin, Win = input.shape
    N, _, _ = theta.shape  # Ensure theta is (N, 2, 3)
    Hout, Wout = size[2:] # Extract output size
    
    # Check output size and theta shape
    assert len(size) == 4 and size[0] == N and size[1] == C, "Output size mismatch"

    # Create output tensor
    output = torch.zeros(size, dtype=input.dtype, device=input.device)
    
    # Create grid
    grid = torch.zeros((N, Hout, Wout, 2), dtype=torch.float32, device=input.device)

    # Define grid launch configuration
    grid_size = (triton.cdiv(N * Hout * Wout, 1024),) # Tune block size as needed
    affine_grid_kernel[grid_size](theta.to(torch.float32), grid, N, C, Hin, Win, Hout, Wout, align_corners)
    
    # Apply grid_sample
    output = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return output
