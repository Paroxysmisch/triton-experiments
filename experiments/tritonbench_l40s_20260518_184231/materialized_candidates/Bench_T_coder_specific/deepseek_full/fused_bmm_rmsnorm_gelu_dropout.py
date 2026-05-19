import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function
from torch.distributed._tensor.utils import _shape_str
from torch.distributed._tensor.local_device import get_local_device
from torch.distributed._tensor.experimental import local_map, local_tensor
from torch.distributed._tensor.experimental import store as dist_store
from torch.distributed._tensor.experimental import (
    PLACEMENT_GROUP,
    make_aware,
    make_uninitialized,
    get_global_size,
    get_global_rank,
    get_local_size,
    get_local_rank,
)
from torch.distributed._tensor.experimental import (
    init_local_tensor,
    init_local_tensor_contiguous,
)
from torch.distributed._tensor.experimental.utils import (
    get_output_buffer,
    to_local_tensor,
)
from torch.distributed._tensor.experimental.replication_config import (
    ReplicationConfig,
)
from torch.distributed._tensor.experimental.meta_tensor import MetaTensor
from torch.distributed._tensor.experimental.ops.dist_op import distribute_ddp, distribute_model
from torch.distributed._tensor.experimental.ops.dist_op import (
    _to_local_tensor,
)
from torch.distributed._tensor.experimental.ops.replicate_ddp import (
    replicate_ddp,
)
from torch.distributed._tensor.experimental.ops.init_ddp import init_ddp
from torch.distributed._tensor.experimental.ops.reduce_ddp import reduce_ddp
from torch.distributed._tensor.experimental.ops.allreduce_ddp import (
    allreduce_ddp,
)
from torch.distributed._tensor.experimental.ops.broadcast_ddp import (
    broadcast_ddp,
)
from torch.distributed._tensor.experimental.ops.all_ddp import all_ddp
from torch.distributed._tensor.experimental.ops.any_ddp import any_ddp
from torch.distributed._tensor.experimental.ops.reduce_scatter_ddp import (
    reduce_scatter_ddp,
)
from torch.distributed._tensor.experimental.ops.gather_ddp import gather_ddp
from torch.distributed._tensor.experimental.ops.barrier_ddp import barrier_ddp
from torch.distributed._tensor.experimental.ops.meta_init_ddp import meta_init_ddp
from torch.distributed._tensor.experimental.ops.meta_allreduce_ddp import (
    meta_allreduce_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_broadcast_ddp import (
    meta_broadcast_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_reduce_ddp import (
    meta_reduce_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_all_ddp import meta_all_ddp
from torch.distributed._tensor.experimental.ops.meta_any_ddp import meta_any_ddp
from torch.distributed._tensor.experimental.ops.meta_reduce_scatter_ddp import (
    meta_reduce_scatter_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_gather_ddp import (
    meta_gather_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_barrier_ddp import (
    meta_barrier_ddp,
)
from torch.distributed._tensor.experimental.ops.local_store_ddp import (
    local_store_ddp,
)
from torch.distributed._tensor.experimental.ops.local_load_ddp import (
    local_load_ddp,
)
from torch.distributed._tensor.experimental.ops.local_reduce_ddp import (
    local_reduce_ddp,
)
from torch.distributed._tensor.experimental.ops.local_allreduce_ddp import (
    local_allreduce_ddp,
)
from torch.distributed._tensor.experimental.ops.local_broadcast_ddp import (
    local_broadcast_ddp,
)
from torch.distributed._tensor.experimental.ops.local_all_ddp import local_all_ddp
from torch.distributed._tensor.experimental.ops.local_any_ddp import local_any_ddp
from torch.distributed._tensor.experimental.ops.local_meta_reduce_scatter_ddp import (
    local_meta_reduce_scatter_ddp,
)
from torch.distributed._tensor.experimental.ops.local_meta_gather_ddp import (
    local_meta_gather_ddp,
)
from torch.distributed._tensor.experimental.ops.local_meta_barrier_ddp import (
    local_meta_barrier_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_reduce_scatter_bucket_size_ddp import (
    meta_reduce_scatter_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_gather_bucket_size_ddp import (
    meta_gather_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_barrier_bucket_size_ddp import (
    meta_barrier_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_reduce_bucket_size_ddp import (
    local_reduce_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_allreduce_bucket_size_ddp import (
    local_allreduce_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_broadcast_bucket_size_ddp import (
    local_broadcast_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_meta_reduce_scatter_bucket_size_ddp import (
    local_meta_reduce_scatter_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_meta_gather_bucket_size_ddp import (
    local_meta_gather_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.local_meta_barrier_bucket_size_ddp import (
    local_meta_barrier_bucket_size_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_reduce_scatter_contiguous_ddp import (
    meta_reduce_scatter_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_gather_contiguous_ddp import (
    meta_gather_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.meta_barrier_contiguous_ddp import (
    meta_barrier_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.local_store_contiguous_ddp import (
    local_store_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.local_load_contiguous_ddp import (
    local_load_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.local_reduce_contiguous_ddp import (
    local_reduce_contiguous_ddp,
)
from torch.distributed._tensor.experimental.ops.local_allreduce_contiguous_ddp import (
    local_allreduce_contigu
