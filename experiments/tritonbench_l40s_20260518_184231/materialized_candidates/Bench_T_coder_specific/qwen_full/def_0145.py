import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel for psi_0
@triton.jit
def psi_0(x):
    # Series expansion for psi_0
    z = 1 / x
    z2 = z * z
    p0 = -2.04032302807999361610816136837999495689
    p1 = 17.71471779056776769434974223782708387338
    p2 = -130.6034684919584829667723217191678042073
    p3 = 799.3667853443463732494604627849650549617
    p4 = -4834.26036513770832598028708861743773318
    p5 = 29928.5856737691169133629973153428987003
    p6 = -182397.3878349161369794714314099849480493
    p7 = 1130312.732979746669294495695199393285148
    p8 = -7141640.093676381276964387946614740114025
    p9 = 46164447.82113179358605296108986937975944
    p10 = -299291441.4438294132792231903621171536439
    p11 = -1.999999999999999105824114791346849253857e-08
    return (p0 + z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * (p5 + z * (p6 + z * (p7 + z * (p8 + z * (p9 + z * (p10 + 11.0 * z * p11)))))))))) / (z2 * (1.0 + z))

# Triton kernel for derivatives of psi
@triton.jit
def derivatives_of_psi(n, x):
    # Recursive calculation for higher derivatives of psi
    if n == 0:
        return psi_0(x)
    else:
        z = 1 / x
        z2 = z * z
        return (-n * derivatives_of_psi(n - 1, x) + psi_0(x) - 0.5 * (n - 1) * (n - 1) * z2) / z2

# Wrapper function for polygamma
def polygamma(n, input, *, out=None) -> torch.Tensor:
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    # Compute polygamma using Triton
    return derivatives_of_psi(n, input)
