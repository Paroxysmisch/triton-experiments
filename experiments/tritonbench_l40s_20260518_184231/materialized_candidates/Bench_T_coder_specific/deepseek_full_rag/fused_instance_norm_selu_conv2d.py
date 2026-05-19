import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ._triton import fused_instance_norm_selu_conv2d as _fused_instance_norm_selu_conv2d
from ._triton import fused_instance_norm_conv2d as _fused_instance_norm_conv2d
from ._triton import fused_selu_conv2d as _fused_selu_conv2d
from ._triton import instance_norm_conv2d as _instance_norm_conv2d
from ._triton import instance_norm_relu_conv2d as _instance_norm_relu_conv2d
from ._triton import selu_conv2d as _selu_conv2d
from ._triton import conv2d as _conv2d
from ._triton import instance_norm_relu_conv2d_dw as _instance_norm_relu_conv2d_dw
from ._triton import instance_norm_relu_conv2d_dw_sl as _instance_norm_relu_conv2d_dw_sl
from ._triton import instance_norm_relu_dw_sl as _instance_norm_relu_dw_sl
from ._triton import instance_norm_relu_sl as _instance_norm_relu_sl
from ._triton import instance_norm_relu_conv2d_sl as _instance_norm_relu_conv2d_sl
from ._triton import instance_norm_relu_conv2d_dw_act as _instance_norm_relu_conv2d_dw_act
from ._triton import instance_norm_relu_conv2d_dw_act_sl as _instance_norm_relu_conv2d_dw_act_sl
from ._triton import instance_norm_relu_dw_act_sl as _instance_norm_relu_dw_act_sl
from ._triton import instance_norm_relu_act_sl as _instance_norm_relu_act_sl
from ._triton import instance_norm_relu_conv2d_act_sl as _instance_norm_relu_conv2d_act_sl
from ._triton import instance_norm_relu_conv2d_dw_act_sl_int8 as _instance_norm_relu_conv2d_dw_act_sl_int8
from ._triton import instance_norm_relu_dw_act_sl_int8 as _instance_norm_relu_dw_act_sl_int8
from ._triton import instance_norm_relu_act_sl_int8 as _instance_norm_relu_act_sl_int8
from ._triton import instance_norm_relu_conv2d_act_sl_int8 as _instance_norm_relu_conv2d_act_sl_int8
from ._triton import fused_instance_norm_relu_conv2d as _fused_instance_norm_relu_conv2d
from ._triton import fused_instance_norm_relu_conv2d_dw as _fused_instance_norm_relu_conv2d_dw
from ._triton import fused_instance_norm_relu_conv2d_dw_sl as _fused_instance_norm_relu_conv2d_dw_sl
from ._triton import fused_instance_norm_relu_dw_sl as _fused_instance_norm_relu_dw_sl
from ._triton import fused_instance_norm_relu_sl as _fused_instance_norm_relu_sl
from ._triton import fused_instance_norm_relu_conv2d_sl as _fused_instance_norm_relu_conv2d_sl
from ._triton import fused_instance_norm_relu_conv2d_dw_act as _fused_instance_norm_relu_conv2d_dw_act
from ._triton import fused_instance_norm_relu_conv2d_dw_act_sl as _fused_instance_norm_relu_conv2d_dw_act_sl
from ._triton import fused_instance_norm_relu_dw_act_sl as _fused_instance_norm_relu_dw_act_sl
from ._triton import fused_instance_norm_relu_act_sl as _fused_instance_norm_relu_act_sl
from ._triton import fused_instance_norm_relu_conv2d_act_sl as _fused_instance_norm_relu_conv2d_act_sl
from ._triton import fused_instance_norm_relu_conv2d_dw_act_sl_int8 as _fused_instance_norm_relu_conv2d_dw_act_sl_int8
from ._triton import fused_instance_norm_relu_dw_act_sl_int8 as _fused_instance_norm_relu_dw_act_sl_int8
from ._triton import fused_instance_norm_relu_act_sl_int8 as _fused_instance_norm_relu_act_sl_int8
from ._triton import fused_instance_norm_relu_conv2d_act_sl_int8 as _fused_instance_norm_relu_conv2d_act_sl_int8

__all__ = [
    "fused_instance_norm_selu_conv2d",
    "fused_instance_norm_conv2d",
    "fused_selu_conv2d",
    "instance_norm_conv2d",
    "instance_norm_relu_conv2d",
    "instance_norm_relu_dw",
    "instance_norm_relu_dw_sl",
    "instance_norm_relu_sl",
    "instance_norm_relu_conv2d_sl",
    "instance_norm_relu_conv2d_dw_act",
    "instance_norm_relu_conv2d_dw_act_sl",
    "instance_norm_relu_dw_act_sl",
    "instance_norm_relu_act_sl",
    "instance_norm_relu_conv2d_act_sl",
    "instance_norm_relu_conv2d_dw_act_sl_int8",
    "instance_norm_relu_dw_act_sl_int8",
    "instance_norm_relu_act_sl_int8",
    "instance_norm_relu_conv2d_act_sl_int8",
    "fused_instance_norm_relu_conv2d",
    "fused_instance_norm_relu_conv2d_dw",
    "fused_instance_norm_relu_conv2d_dw_sl",
    "fused_instance_norm_relu_dw_sl",
    "fused_instance_norm_relu_sl",
    "fused_instance_norm_relu_conv2d_sl",
    "fused_instance_norm_relu_conv2d_dw_act",
    "fused_instance_norm_relu_conv2d_dw_act_sl",
    "fused_instance_norm_relu_dw_act_sl",
    "fused_instance_norm_relu_act_sl",
    "fused_instance_norm_relu_conv2d_act_sl",
    "fused_instance_norm_relu_conv2d_dw_act_sl_int8",
    "fused_instance_norm_relu_dw_act_sl_int8",
    "fused_instance_norm_relu_act_sl_int8",
    "fused_instance_norm_relu_conv2d_act_sl_int8",
]


class _FusedInstance
