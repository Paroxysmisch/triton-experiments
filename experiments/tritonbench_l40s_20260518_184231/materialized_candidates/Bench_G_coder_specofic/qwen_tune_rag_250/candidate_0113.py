import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
from fla.utils import contiguous

@triton.jit
def triton_red_fused_native_layer_norm_0(in0, in1, in2, in3, in4, out_ptr0, out_ptr1, buf0, buf3, buf4, tmp3_mean, tmp3_m2, tmp3_weight, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = rbase
    tmp5 = tmp4 < rnumel
    tmp6 = tmp4
    tmp9 = tl.load(in2 + (tmp6), None).to(tl.float32)
    tmp13 = tmp5.to(tl.int64) * 1
    tmp15 = xmask.to(tl.int64) * 1
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 == 2
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tmp18 + 1
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sum(tmp9, 0) / tmp20
    tmp22 = tmp9 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp5.to(tl.int64) * 1
    tmp26 = xmask.to(tl.int64) * 1
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 == 2
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29 + 1
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.sum(tmp23, 0) / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = tl.math.rsqrt(tmp34)
    tl.store(tmp3_mean + (x0), tmp21, None)
    tl.store(tmp3_weight + (x0), tmp35, None)
    for i in range(0, 1):
        rindex = rbase + i
        rmask = rindex < rnumel
        r1 = rindex
        tmp37 = x0
        tmp38 = r1
        tmp39 = tmp2
        tmp40 = tmp22
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp38 < rnumel
        tmp43 = tl.load(in3 + (tmp38), None).to(tl.float32)
        tmp44 = tmp42.to(tl.int64) * 1
        tmp45 = 1
        tmp46 = tmp44 * tmp45
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp43 * tmp47
        tmp49 = tmp41 * tmp49
        tmp50 = tmp48 + tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp51
        tmp53 = tmp39 - tmp21
        tmp54 = tmp52 * tmp53
        tmp55 = 1.0
        tmp56 = tmp54 * tmp55
        tmp57 = tmp56 * tmp35
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 * tmp39
        tmp60 = tmp6 + i
        tl.store(buf0 + (tmp59), tmp60, None)
        tmp61 = tmp60.to(tl.int64)
        tmp62 = tmp61 * 1
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp63 + tmp39
        tmp65 = tmp64.to(tl.float32)
        tmp66 = tmp65 * tmp35
        tmp67 = tmp66.to(tl.float32)
        tmp68 = 0
        tmp69 = tmp67 + tmp68
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tmp70 + tmp58
        tmp72 = tmp71.to(tl.float32)
        tmp73 = tmp72 + tmp56
        tmp74 = tmp73.to(tl.float32)
        tmp75 = tmp74 + tmp50
        tmp76 = tmp75.to(tl.float32)
        tmp77 = tmp76 * 0.5
        tmp78 = tmp77.to(tl.float32)
        tmp79 = tmp78 + tmp39
        tmp80 = tmp79.to(tl.float32)
        tmp81 = tmp80 * tmp35
        tmp82 = tmp81.to(tl.float32)
        tmp83 = tmp82 * 2
        tmp84 = tmp83.to(tl.float32)
        tmp85 = tmp84 + tmp39
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp86 * tmp35
        tmp88 = tmp87.to(tl.float32)
        tmp89 = tmp88 + tmp39
        tmp90 = tmp89.to(tl.float32)
        tmp91 = tmp90 * tmp35
        tmp92 = tmp91.to(tl.float32)
        tmp93 = tmp92 + tmp39
        tmp94 = tmp93.to(tl.float32)
        tmp95 = tmp94 * tmp35
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96 + tmp39
        tmp98 = tmp97.to(tl.float32)
        tmp99 = tmp98 * tmp35
        tmp100 = tmp99.to(tl.float32)
        tmp101 = tmp100 + tmp39
        tmp102 = tmp101.to(tl.float32)
        tmp103 = tmp102 * tmp35
        tmp104 = tmp103.to(tl.float32)
        tmp105 = tmp104 + tmp39
        tmp106 = tmp105.to(tl.float32)
        tmp107 = tmp106 * tmp35
        tmp108 = tmp107.to(tl.float32)
        tmp109 = tmp108 + tmp39
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp110 * tmp35
        tmp112 = tmp111.to(tl.float32)
        tmp113 = tmp112 + tmp39
        tmp114 = tmp113.to(tl.float32)
        tmp115 = tmp114 * tmp35
        tmp116 = tmp115.to(tl.float32)
        tmp117 = tmp116 + tmp39
        tmp118 = tmp117.to(tl.float32)
        tmp119 = tmp118 * tmp35
        tmp120 = tmp119.to(tl.float32)
        tmp121 = tmp120 + tmp39
        tmp122 = tmp121.to(tl.float32)
        tmp123 = tmp122 * tmp35
        tmp124 = tmp123.to(tl.float32)
        tmp125 = tmp124 + tmp39
        tmp126 = tmp125.to(tl.float32)
        tmp127 = tmp126 * tmp35
        tmp128 = tmp127.to(tl.float32)
        tmp129 = tmp128 + tmp39
        tmp130 = tmp129.to(tl.float32)
        tmp131 = tmp130 * tmp35
        tmp132 = tmp131.to(tl.float32
