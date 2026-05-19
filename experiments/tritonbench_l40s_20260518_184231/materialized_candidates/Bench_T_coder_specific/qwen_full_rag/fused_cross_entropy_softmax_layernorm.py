import logging
import math
import warnings

import torch
from packaging import version

from ..utils import ext_loader

try:
    from torch._inductor.select_algorithm import extern_kernels
except ImportError:
    try:
        from torch._inductor.utils import extern_kernels
    except ImportError:
        from torch._inductor import extern_kernels

try:
    from torch._inductor.triton_heuristics import pointwise
except ImportError:
    try:
        from torch._inductor.heuristics import pointwise
    except ImportError:
        from torch._inductor import heuristics as pointwise

try:
    from torch._inductor.triton_heuristics import reduction
except ImportError:
    try:
        from torch._inductor.heuristics import reduction
    except ImportError:
        from torch._inductor import heuristics as reduction

try:
    from torch._inductor.triton_heuristics import persistent_reduction
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_reduction
    except ImportError:
        from torch._inductor import heuristics as persistent_reduction

try:
    from torch._inductor.triton_heuristics import scan
except ImportError:
    try:
        from torch._inductor.heuristics import scan
    except ImportError:
        from torch._inductor import heuristics as scan

try:
    from torch._inductor.triton_heuristics import pipeline
except ImportError:
    try:
        from torch._inductor.heuristics import pipeline
    except ImportError:
        from torch._inductor import heuristics as pipeline

try:
    from torch._inductor.triton_heuristics import persistent_scan
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_scan
    except ImportError:
        from torch._inductor import heuristics as persistent_scan

try:
    from torch._inductor.triton_heuristics import threestage
except ImportError:
    try:
        from torch._inductor.heuristics import threestage
    except ImportError:
        from torch._inductor import heuristics as threestage

try:
    from torch._inductor.triton_heuristics import persistent_threestage
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_threestage
    except ImportError:
        from torch._inductor import heuristics as persistent_threestage

try:
    from torch._inductor.triton_heuristics import persistent_grid
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_grid
    except ImportError:
        from torch._inductor import heuristics as persistent_grid

try:
    from torch._inductor.triton_heuristics import warps_kernel
except ImportError:
    try:
        from torch._inductor.heuristics import warps_kernel
    except ImportError:
        from torch._inductor import heuristics as warps_kernel

try:
    from torch._inductor.triton_heuristics import vnni
except ImportError:
    try:
        from torch._inductor.heuristics import vnni
    except ImportError:
        from torch._inductor import heuristics as vnni

try:
    from torch._inductor.triton_heuristics import w4a8
except ImportError:
    try:
        from torch._inductor.heuristics import w4a8
    except ImportError:
        from torch._inductor import heuristics as w4a8

try:
    from torch._inductor.triton_heuristics import w4g8
except ImportError:
    try:
        from torch._inductor.heuristics import w4g8
    except ImportError:
        from torch._inductor import heuristics as w4g8

try:
    from torch._inductor.triton_heuristics import w8a8
except ImportError:
    try:
        from torch._inductor.heuristics import w8a8
    except ImportError:
        from torch._inductor import heuristics as w8a8

try:
    from torch._inductor.triton_heuristics import w8g8
except ImportError:
    try:
        from torch._inductor.heuristics import w8g8
    except ImportError:
        from torch._inductor import heuristics as w8g8

try:
    from torch._inductor.triton_heuristics import w4_float_g8_mixed_a
except ImportError:
    try:
        from torch._inductor.heuristics import w4_float_g8_mixed_a
    except ImportError:
        from torch._inductor import heuristics as w4_float_g8_mixed_a

try:
    from torch._inductor.triton_heuristics import w4_float_g8_mixed_b
except ImportError:
    try:
        from torch._inductor.heuristics import w4_float_g8_mixed_b
    except ImportError:
        from torch._inductor import heuristics as w4_float_g8_mixed_b

try:
    from torch._inductor.triton_heuristics import custom_grid
except ImportError:
    try:
        from torch._inductor.heuristics import custom_grid
    except ImportError:
        from torch._inductor import heuristics as custom_grid

try:
    from torch._inductor.triton_heuristics import persistent_custom_grid
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_custom_grid
    except ImportError:
        from torch._inductor import heuristics as persistent_custom_grid

try:
    from torch._inductor.triton_heuristics import triton_heuristics
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics

try:
    from torch._inductor.triton_heuristics import triton_heuristics_with_attention
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics_with_attention
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics_with_attention

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics_with_attention
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics_with_attention
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics_with_attention

try:
    from torch._inductor.triton_heuristics import triton_heuristics_with_grouping
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics_with_grouping
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics_with_grouping

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics_with_grouping
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics_with_grouping
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics_with_grouping

try:
    from torch._inductor.triton_heuristics import triton_heuristics_with_modulation
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics_with_modulation
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics_with_modulation

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics_with_modulation
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics_with_modulation
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics_with_modulation

try:
    from torch._inductor.triton_heuristics import triton_heuristics_with_weight_only_matmul
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics_with_weight_only_matmul
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics_with_weight_only_matmul

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics_with_weight_only_matmul
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics_with_weight_only_matmul
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics_with_weight_only_matmul

try:
    from torch._inductor.triton_heuristics import triton_heuristics_with_flash_attn
except ImportError:
    try:
        from torch._inductor.heuristics import triton_heuristics_with_flash_attn
    except ImportError:
        from torch._inductor import heuristics as triton_heuristics_with_flash_attn

try:
    from torch._inductor.triton_heuristics import persistent_triton_heuristics_with_flash_attn
except ImportError:
    try:
        from torch._inductor.heuristics import persistent_triton_heuristics_with_flash_attn
    except ImportError:
        from torch._inductor import heuristics as persistent_triton_heuristics_with_flash_attn
