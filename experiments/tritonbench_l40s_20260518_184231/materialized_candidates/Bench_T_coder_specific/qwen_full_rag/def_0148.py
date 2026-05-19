import torch
import triton
import triton.language as tl
from flag_gems.utils.libgem import libgem

@triton.jit(do_not_specialize=["n_elements", "step"])
def _adam_fused_kernel(
    params_ptr,
    grad_ptr,
    mom1_ptr,
    mom2_ptr,
    lr,
    beta1,
    beta2,
    eps,
    wd,
    n_elements,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Offsetted pointers
    offset_params_ptr = params_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_mom1_ptr = mom1_ptr + offsets
    offset_mom2_ptr = mom2_ptr + offsets

    # Load
    param = tl.load(offset_params_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    mom1 = tl.load(offset_mom1_ptr, mask=mask)
    mom2 = tl.load(offset_mom2_ptr, mask=mask)

    # Stepweight decay
    param = param * (1 - lr * wd)

    # Diff between momentum running average and grad
    diff = mom1 - grad

    # Weight update
    update = diff * (1 - beta1)

    # Torch.sign
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    # Parameter momentum running average update
    new_param = param + update_sign

    # Clamps grad norm
    grad_norm = tl.minimum(tl.abs(grad), 1)
    diff_norm = tl.minimum(tl.abs(diff), 1)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # Momentum 1 update
    new_mom1 = diff * bias_correction1 / (1 - beta1) + grad

    # Momentum 2 update
    new_mom2 = (
        mom2 * bias_correction2 / (1 - beta2) + (grad_norm * grad_norm) * (1 - beta2)
    )

    # Weight update
    diff_over_dif = diff / new_mom2

    # Square root
    div = diff_over_dif / tl.sqrt(new_mom2)

    # Rounding trick for div
    div_approx = tl.where(div <= 0, tl.ceil(div - 0.5), tl.floor(div + 0.5))
    sqrt_new_mom2 = new_mom2 + div_approx

    # Numerical issues check
    inf_mask = sqrt_new_mom2 == float("inf")
    sqrt_new_mom2 = tl.where(inf_mask, new_mom2, sqrt_new_mom2)

    # Final parameter update
    ret = new_param - update_sign * (lr / (eps + tl.sqrt(sqrt_new_mom2)))

    # Store new momentum running averages
    tl.store(offset_mom1_ptr, new_mom1, mask=mask)
    tl.store(offset_mom2_ptr, new_mom2, mask=mask)

    # Store new params
    tl.store(offset_params_ptr, ret.to(param.dtype), mask=mask)


class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        fused=True,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

        self.fused = fused
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable

        self.num_groups = len(self.param_groups)
        self._state["foreach"] = foreach

    @property
    def state_names(self):
        return ["exp_avg", "exp_avg_sq"]

    def _create_state_for_param_group(self, group_idx, param_group):
        n_tensors = 2 if not param_group["amsgrad"] else 3
        self.state[group_idx] = [
            {},
        ] * n_tensors

    def _get_state_for_parameter(self, group_idx, param):
        return self.state[group_idx][param]

    def _set_state_for_parameter(self, group_idx, param, state):
        self.state[group_idx][param] = state

    def step(self, closure=None):
        has_flat_params = all(
            isinstance(p, torch.Tensor) for p in self._params_with_grad()
        )
        if has_flat_params:
            return self._step(closure=closure)
        else:
            return self._step_tensorwise(closure=closure)

    def _step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, param_group in enumerate(self.param_groups):
            params_with_grad = self._params_with_grad(group_idx)
            exp_avgs = [self._get_exp_avg(g) for g in params_with_grad]
            exp_avg_sqs = [self._get_exp_avg_sq(g) for g in params_with_grad]
            old_grads = [self._get_grad(g) for g in params_with_grad]
            new_grads = [None for g in params_with_grad]
            f_weights = [self._get_weight(g) for g in params_with_grad]
            new_weights = [None for g in params_with_grad]
            lr = param_group["lr"]
            beta1, beta2 = param_group["betas"]
            eps = param_group["eps"]
            wd = param_group["weight_decay"]
            amsgrad = param_group["amsgrad"]
            n_tensors = 2 if not amsgrad else 3
            n_elements = sum(p.numel() for p in params_with_grad)

            step = (
                self.state[self.num_groups - 1][params_with_grad[0]]["step"]
                if group_idx == self.num_groups - 1
                else self.state[group_idx][params_with_grad[0]]["step"]
            )
            step += 1

            if not self.fused:
                for i, (params, exp_avg, exp_avg_sq, old_grad, new_grad, f_weight, new_weight) in enumerate(zip(
                        params_with_grad, exp_avgs, exp_avg_sqs, old_grads, new_grads, f_weights, new_weights)):
                    self._update.Adam(i, n_tensors, params, exp_avg, exp_avg_sq, old_grad, new_grad, f_weight,
                                      new_weight, lr, beta1, beta2, eps, wd, amsgrad, n_elements, step)
            else:
                grad_sum = self._create_param_if_needed(n_tensors, "grad", group_idx, n_elements)
                weight_sum = self._create_param_if_needed(n_tensors, "weight", group_idx, n_elements)
                buf = self._create_buffer_if_needed("adam_buf", group_idx, n_elements)

                # Copy gradients
                buf.copy_(torch.stack(old_grads).to(torch.float32))
                buf.mul_(beta1).add_(torch.stack(new_grads).to(torch.float32), alpha=1 - beta1)

                # Update gradients norm
                grad_norm = torch.sqrt((buf * buf).sum()) / math.sqrt(n_elements)

                # Weight update
                buf.copy_(torch.stack(f_weights))
                buf.mul_(1 - lr * wd)

                # Accumulates momentum running average
                torch.maximum(buf, buf * beta1 + torch.stack(new_grads).to(torch.float32), out=grad_sum)

                # Decay the momentum running average coefficient
                buf.copy_(grad_sum)
                buf.div_(1 - beta1 ** step)

                # Accumulates
