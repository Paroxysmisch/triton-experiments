import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import rsqrt
from .utils import calculate_settings

@triton.jit
def affine_grid(theta, size, align_corners):
    grid = tl.zeros((size[0], size[1], size[2], 3), dtype=tl.float32)
    theta = theta.to(tl.float32)
    x = tl.arange(0, size[3]).to(tl.float32)
    y = tl.arange(0, size[2]).to(tl.float32)
    if not align_corners:
        x = (x + 0.5) / size[3] * 2 - 1
        y = (y + 0.5) / size[2] * 2 - 1
    else:
        x = x / (size[3] - 1) * 2 - 1
        y = y / (size[2] - 1) * 2 - 1
    grid_x = tl.broadcast_to(tl.reshape(x, [1, 1, size[3]], grid.dtype), [size[2], size[3], 3])
    grid_y = tl.broadcast_to(tl.reshape(y, [1, size[2], 1], grid.dtype), [size[2], size[3], 3])
    grid = grid + grid_x
    grid = grid + grid_y
    theta = tl.reshape(theta, [size[0], size[1], 6])
    N = size[0] * size[1]
    batch_index = tl.arange(0, N) // size[1]
    theta_batch = tl.gather(theta, batch_index, 0)
    V = tl.matmul(theta_batch, grid)
    return V

@triton.jit
def trilinear_interpolate(
        img,
        x,
        y,
        z,
        padding_mode: str,
        C: int,
        H: int,
        W: int,
        D: int,
        grad_scale: float,
        mode: str,
):
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    z0 = tl.floor(z).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0 = tl.maximum(x0, 0)
    y0 = tl.maximum(y0, 0)
    z0 = tl.maximum(z0, 0)
    x1 = tl.minimum(x1, W - 1)
    y1 = tl.minimum(y1, H - 1)
    z1 = tl.minimum(z1, D - 1)
    w_x = x - x0
    w_y = y - y0
    w_z = z - z0
    if padding_mode == 'zeros':
        w_x = tl.where(x0 < W, w_x, 0)
        w_y = tl.where(y0 < H, w_y, 0)
        w_z = tl.where(z0 < D, w_z, 0)
    elif padding_mode == 'border':
        w_x = tl.minimum(w_x, 1)
        w_x = tl.maximum(w_x, 0)
        w_y = tl.minimum(w_y, 1)
        w_y = tl.maximum(w_y, 0)
        w_z = tl.minimum(w_z, 1)
        w_z = tl.maximum(w_z, 0)
    elif padding_mode == 'reflection':
        w_x = tl.where(x0 < W, w_x, 2 * (W - 1 - x0) - w_x)
        w_y = tl.where(y0 < H, w_y, 2 * (H - 1 - y0) - w_y)
        w_z = tl.where(z0 < D, w_z, 2 * (D - 1 - z0) - w_z)
        w_x = tl.minimum(w_x, 1)
        w_x = tl.maximum(w_x, 0)
        w_y = tl.minimum(w_y, 1)
        w_y = tl.maximum(w_y, 0)
        w_z = tl.minimum(w_z, 1)
        w_z = tl.maximum(w_z, 0)
    w_x = w_x[:, None]
    w_y = w_y[:, None]
    w_z = w_z[:, None]
    i_x0 = x0 + W * y0 + W * H * z0
    i_x1 = x1 + W * y0 + W * H * z0
    i_y0 = x0 + W * y0 + W * H * z1
    i_y1 = x0 + W * y1 + W * H * z0
    i_z0 = x0 + W * y0 + W * H * z0
    i_z1 = x0 + W * y1 + W * H * z1
    img0 = tl.load(img + i_x0 + tl.arange(0, C), mask=i_x0 < W * H * D, other=0)
    img1 = tl.load(img + i_x1 + tl.arange(0, C), mask=i_x1 < W * H * D, other=0)
    img2 = tl.load(img + i_y0 + tl.arange(0, C), mask=i_y0 < W * H * D, other=0)
    img3 = tl.load(img + i_y1 + tl.arange(0, C), mask=i_y1 < W * H * D, other=0)
    img4 = tl.load(img + i_z0 + tl.arange(0, C), mask=i_z0 < W * H * D, other=0)
    img5 = tl.load(img + i_z1 + tl.arange(0, C), mask=i_z1 < W * H * D, other=0)
    img0 = img0 * (1 - w_z) + img4 * w_z
    img1 = img1 * (1 - w_z) + img5 * w_z
    img2 = img2 * (1 - w_z) + img6 * w_z
    img3 = img3 * (1 - w_z) + img7 * w_z
    img0 = img0 * (1 - w_y) + img2 * w_y
    img1 = img1 * (1 - w_y) + img3 * w_y
    if mode == 'bilinear':
        img0 = img0 * (1 - w_x) + img1 * w_x
        img1 = img2 * (1 - w_x) + img3 * w_x
    return img0 + grad_scale, img1 + grad_scale

@triton.jit
def trilinear_sampler(img, grid, padding_mode, align_corners, C, H, W, D, mode, grad_scale):
    if not align_corners:
        grid = grid / (D - 1) * 2 - 1
    x = grid[:, :, 0]
    y = grid[:, :, 1]
    z = grid[:, :, 2]
    return trilinear_interpolate(img, x, y, z, padding_mode, C, H, W, D, grad_scale, mode)

@triton.jit
def triton_grid_sample(input, grid, theta, size, padding_mode, align_corners, mode, grad_scale):
    N = size[0] * size[1]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    D = grid.shape[3]
    input = input.reshape(N, C, H, W)
    grid = grid.reshape(N, H, W, D)
    output = trilinear_sampler(input, grid, padding_mode, align_corners, C, H, W, D, mode, grad_scale)
    output = output.reshape(size)
    return output

def grid_sample_with_affine_triton(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    grid = affine_grid(theta, size, align_corners)
    return triton_grid_sample(input, grid, theta, size, padding_mode, align_corners, mode, 0)
