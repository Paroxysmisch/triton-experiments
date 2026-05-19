import torch
import triton
import triton.language as tl

# Assuming the smaller Triton kernels from the provided document (e.g., calc_mean_and_inv_std,
# update_ema, standardize, and apply_act_func) are already imported or defined in the same module.
# For example:
#
# from .some_module import calc_mean_and_inv_std, standardize, update_ema
# from .act_kernels import apply_act_func

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    """
    Applies Batch Normalization followed by the Hardsigmoid activation function
    on the input tensor x. When training=True, this updates running_mean and
    running_var in place using exponential moving averages of mean and variance.
    
    Args:
        x (Tensor): Input tensor for batch normalization and activation.
        running_mean (Tensor): The running mean buffer (persistent).
        running_var (Tensor): The running variance buffer (persistent).
        weight (Tensor, optional): Learnable weight of size C for the normalized tensor. Default: None
        bias (Tensor, optional): Learnable bias of size C for the normalized tensor. Default: None
        training (bool, optional): Flag for training mode, used to update running estimates. Default: False
        momentum (float, optional): The value for the running mean and variance momentum. Default: 0.1
        eps (float, optional): Small constant added to variance to improve numerical stability. Default: 1e-5
        inplace (bool, optional): If True, perform Hardsigmoid in-place. Default: False
    
    Returns:
        Tensor: Output tensor after batch normalization and Hardsigmoid activation.
    """
    # Ensure weight and bias are not None
    # If not provided, default to ones and zeros for standard BN
    if weight is None:
        weight = torch.ones_like(running_mean, dtype=x.dtype, device=x.device)
    if bias is None:
        bias = torch.zeros_like(running_mean, dtype=x.dtype, device=x.device)

    # Reshape x to 2D [N, C'] or assume the last dimension is feature dimension
    # for computing per-feature statistics. This is a simplistic approach.
    original_shape = x.shape
    numel = x.numel()

    # For demonstration, flatten all but the last dimension of x into N, then treat last as features
    # Adjust this as needed for your specific BN dimensionality.
    if x.dim() > 2:
        x_2d = x.view(-1, original_shape[-1])
        last_dim = x_2d.shape[1]
    else:
        x_2d = x
        last_dim = x.shape[1] if x.dim() == 2 else x.shape[0]

    # Allocate output same as x
    out_2d = torch.empty_like(x_2d)

    # 1) Compute mean and inverse standard deviation of x_2d
    # Using the provided Triton kernel: calc_mean_and_inv_std
    # Launch one block per row for the simplified example
    grid = (x_2d.shape[0],)
    mean = torch.empty((x_2d.shape[0],), dtype=torch.float32, device=x.device)
    inv_std = torch.empty((x_2d.shape[0],), dtype=torch.float32, device=x.device)
    last_dim_mask = torch.ones((last_dim,), dtype=torch.bool, device=x.device)

    calc_mean_and_inv_std[grid](
        x_2d, 
        last_dim, 
        eps,
        last_dim_mask,
        out_ptr0=mean,
        out_ptr1=inv_std
    )

    # 2) Compute global mean/variance across the batch if training=True
    #    Then update running stats
    if training:
        # The next two lines average the per-row stats to get global mean/var
        batch_mean = mean.mean()
        # var = mean of (1/(inv_std^2)) minus here we read per-row inv_std
        variance = (1. / (inv_std * inv_std)).mean()

        # Update running_mean, running_var in place
        # Using the provided Triton function: update_ema for each
        # We'll do an elementwise approach for demonstration; if multiple channels exist,
        # you would typically store and update them channel-wise.
        # Here we treat running_mean and running_var as size [C], but for a single feature
        # we do the simple approach:
        new_mean = torch.empty_like(running_mean)
        new_var = torch.empty_like(running_var)
        batch_mean_const = torch.tensor(batch_mean, dtype=x.dtype, device=x.device)
        batch_var_const = torch.tensor(variance, dtype=x.dtype, device=x.device)
        count_grid = (running_mean.numel(),)

        update_ema[count_grid](
            running_mean,
            batch_mean_const,
            momentum,
            out_ptr0=new_mean
        )
        update_ema[count_grid](
            running_var,
            batch_var_const,
            momentum,
            out_ptr0=new_var
        )
        running_mean.copy_(new_mean)
        running_var.copy_(new_var)

        # For standardization, we broadcast batch_mean & sqrt(var) across rows
        # so we can reuse the per-row approach
        mean = mean - mean + batch_mean  # shift each row's mean to global mean
        inv_std = inv_std * 0.0 + 1. / torch.sqrt(batch_var_const + eps)
    else:
        # If not training, we use the existing running_mean, running_var
        # We'll standardize using the persistent stats
        # Flatten them to broadcast if needed
        run_inv_std = 1.0 / torch.sqrt(running_var + eps)
        batch_mean_const = running_mean
        # Expand to match each row
        mean = mean * 0.0 + batch_mean_const
        inv_std = inv_std * 0.0 + run_inv_std

    # 3) Standardize x_2d using the BN stats, then update out_2d
    # Using the provided Triton kernel: standardize
    # We'll treat weight, bias as broadcastable as well
    # Flatten weight, bias if needed
    if weight.dim() > 1:
        weight_flat = weight.view(-1)
    else:
        weight_flat = weight
    if bias.dim() > 1:
        bias_flat = bias.view(-1)
    else:
        bias_flat = bias

    standardize_grid = (x_2d.shape[0],)
    standardize[standardize_grid](
        x_2d, 
        mean, 
        inv_std, 
        weight_flat, 
        bias_flat,
        out_ptr0=out_2d
    )

    # 4) Apply Hardsigmoid activation to out_2d
    # Using the provided Triton kernel: apply_act_func
    # E.g., apply_act_func(input, None, None, None, param, 'hardsigmoid', False)
    act_grid = (out_2d.shape[0],)
    # The kernel might produce in-place if requested or create a copy
    # We'll handle the in-place logic in Python for clarity:
    if inplace:
        apply_act_func[act_grid](
            out_2d, None, None, None, 0.0, 'hardsigmoid', True, out_ptr0=out_2d
        )
        y_2d = out_2d
    else:
        y_2d = torch.empty_like(out_2d)
        apply_act_func[act_grid](
            out_2d, None, None, None, 0.0, 'hardsigmoid', False, out_ptr0=y_2d
        )

    # 5) Reshape back to original shape
    y = y_2d.view(original_shape)

    return y
