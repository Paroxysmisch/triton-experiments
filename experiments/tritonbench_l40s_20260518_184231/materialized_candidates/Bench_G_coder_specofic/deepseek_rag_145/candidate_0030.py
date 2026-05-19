@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,
    in_out_ptr1,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
   # the kernel implementation goes here

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    S, D = primals_3.shape
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf1 = buf0
        del buf0  # reuse
        buf2 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf3 = reinterpret_tensor(buf2, (S, 1), (1, 1), 0)
        del buf2  # reuse
        buf4 = empty_strided_cuda((S, D), (D, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            buf1, buf3, primals_3, primals_1, primals_2, buf4, S, D
        )
    return (
        buf4,
        primals_3,
        buf1,
        buf3,
    )
