import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_fwd

@triton.jit
def conv2d_forward_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    conv_param,
    x_batch_stride,
    x_in_feat_stride,
    x_height_stride,
    x_width_stride,
    w_out_feat_stride,
    w_in_feat_stride,
    w_height_stride,
    w_width_stride,
    z_batch_stride,
    z_out_feat_stride,
    z_height_stride,
    z_width_stride,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
    FP16: tl.constexpr,
    TF32: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
):
    # Implementation details omitted for brevity
    pass

class Conv2dFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: int,
        padding: int,
        groups: int,
        benchmark: bool,
        deterministic: bool,
        conv_param: ConvParam,
        compute_dtype: torch.dtype,
        int_dtype: torch.dtype,
    ) -> Tensor:
        assert (
            x.dtype == weight.dtype
        ), "Input and weight should have the same dtype, but got {} and {}".format(
            x.dtype, weight.dtype
        )
        assert bias.dtype == weight.dtype, "Bias should have the same dtype as weight"
        assert (
            x.is_contiguous()
        ), "Input tensor must be contiguous as Triton backend is not friendly to non-contiguous inputs"

        x_shape = x.shape
        x_batch_size, x_in_feats, x_height, x_width = x_shape
        w_out_feats, w_in_feats, w_height, w_width = weight.shape

        assert (
            x_in_feats % groups == 0 and w_out_feats % groups == 0
        ), "Incompatible numbers of input channels and output channels"

        z_batch_size = x_batch_size
        z_out_feats = w_out_feats
        z_height = (x_height - (w_height - 1) - 2 * padding) // stride + 1
        z_width = (x_width - (w_width - 1) - 2 * padding) // stride + 1

        z = torch.empty(
            z_batch_size, z_out_feats, z_height, z_width, device=x.device, dtype=x.dtype
        )
        x_strides = x.stride()
        w_strides = weight.stride()
        z_strides = z.stride()

        x_stride_batch = x_strides[0]
        x_stride_in_feat = x_strides[1]
        x_stride_height = x_strides[2]
        x_stride_width = x_strides[3]

        w_stride_out_feat = w_strides[0]
        w_stride_in_feat = w_strides[1]
        w_stride_height = w_strides[2]
        w_stride_width = w_strides[3]

        z_stride_batch = z_strides[0]
        z_stride_out_feat = z_strides[1]
        z_stride_height = z_strides[2]
        z_stride_width = z_strides[3]

        kernel_size = w_height
        stride = stride
        padding = padding

        FP16 = x.dtype == torch.float16
        TF32 = False

        grid_fn = lambda meta: (
            triton.cdiv(x_batch_size, meta["BLOCK_SIZE_BATCH"]),
            triton.cdiv(x_in_feats, meta["BLOCK_SIZE_IN_FEAT"]),
            triton.cdiv(w_out_feats, meta["BLOCK_SIZE_OUT_FEAT"]),
        )

        def check_config(config):
            _, _, out_feats_per_tile = config
            return out_feats_per_tile >= conv_param.min_out_feat_per_tile

        with torch.cuda.device(x.device.index):
            conv2d_forward_kernel[grid_fn](
                x_ptr=x,
                w_ptr=weight,
                z_ptr=z,
                conv_param=conv_param,
                x_batch_stride=x_stride_batch,
                x_in_feat_stride=x_stride_in_feat,
                x_height_stride=x_stride_height,
                x_width_stride=x_stride_width,
                w_out_feat_stride=w_stride_out_feat,
                w_in_feat_stride=w_stride_in_feat,
                w_height_stride=w_stride_height,
                w_width_stride=w_stride_width,
                z_batch_stride=z_stride_batch,
                z_out_feat_stride=z_stride_out_feat,
                z_height_stride=z_stride_height,
                z_width_stride=z_stride_width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                FP16=FP16,
                TF32=TF32,
                BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
                BLOCK_SIZE_IN_FEAT=BLOCK_SIZE_IN_FEAT,
                BLOCK_SIZE_OUT_FEAT=BLOCK_SIZE_OUT_FEAT,
                num_warps=conv_param.num_warps,
                num_stages=conv_param.num_stages,
                check_config=check_config,
            )

        if bias is not None:
            z += bias[None, :, None, None]

        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.conv_param = conv_param
        ctx.compute_dtype = compute_dtype
        ctx.int_dtype = int_dtype

        return z

def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: Union[int, List[int]],
    padding: Union[int, List[int]],
    groups: int = 1,
    benchmark: bool = False,
    deterministic: bool = False,
) -> Tensor:
    assert (
        x.dtype == weight.dtype
    ), "Input and weight should have the same dtype, but got {} and {}".format(
        x.dtype, weight.dtype
    )
    assert (
        bias.dtype == weight.dtype
    ), "Bias should have the same dtype as weight, but got {} and {}".format(
        bias.dtype, weight.dtype
    )
    assert (
        x.is_contiguous()
    ), "Input tensor must be contiguous as Triton backend is not
