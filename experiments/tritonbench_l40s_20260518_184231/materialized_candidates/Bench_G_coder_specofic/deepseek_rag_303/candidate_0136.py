import torch
import triton
import triton.language as tl
from .ops import CrossEntropyLoss, cross_entropy_loss


def test_cross_entropy_loss():
    world_size = 2
    rank = 0
    torch.manual_seed(42)
    smoothing = 0.0
    logit_scale = 1.0
    lse_square_scale = 0.0
    num_classes = 100
    total_classes = num_classes

    logits = torch.empty(
        (10, num_classes), dtype=torch.float16, device="cuda"
    )  # (num_chunks, num_classes)
    labels = torch.randint(0, num_classes, (10,), dtype=torch.int32, device="cuda")
    ignored_index = -100

    loss_module = CrossEntropyLoss(
        smoothing=smoothing,
        ignore_index=ignored_index,
        logit_scaler=logit_scale,
        loscean=lse_square_scale,
        world_size=world_size,
        rank=rank,
        parallel_mode="tensor",
    )

    loss_module.set_class_ranks_and_counts(
        torch.arange(rank * 50, rank * 50 + num_classes), total_classes
    )

    triton_loss, triton_z_loss = cross_entropy_loss(logits, labels, loss_module)
    torch_loss, _ = torch.nn.CrossEntropyLoss(
        ignore_index=ignored_index, reduction="none"
    )(logits, labels)
    assert torch.allclose(torch_loss, triton_loss)
