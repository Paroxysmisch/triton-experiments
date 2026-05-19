import torch
import triton
from loguru import logger

from vllm.model.triton_kernels.base import fwd, get_block

def decoder_partial(
    hidden_state,
    cell_state,
    output,
    decoder_extended_attention_mask,
    encoder_hidden_states,
    decoder_hidden_states,
    logit_weights,
    prompt_lengths,
    logit_bias,
    args,
    positions,
    timestep,
    init_state,
    first_kv_input = None,
    final_kv_input = None,
    ffn_intermediate_init = None,
    logits_penalty_mask: torch.Tensor = None,
    logits_penalty_freq: torch.Tensor = None,
    logits_penalty_presence: torch.Tensor = None,
    logits_penalty_counts: torch.Tensor = None,
    logits_penalty_cumsum_seq_len: torch.Tensor = None,
):
    batch_size, head_num = logit_weights.shape[:2]
    seqlen = output.size(1)
    assert positions.size(0) == batch_size
    assert logit_weights.dim() == 3
    assert logit_bias.dim() == 2
    assert logit_bias.size(0) == head_num
    assert hidden_state.size() == (
        batch_size,
        seqlen,
        512,
    )  # TODO: The condition should be derived from model config
    assert hidden_state.stride(-1) == 1
    assert hidden_state.stride(-2) == 4096
    assert encoder_hidden_states.size(2) == 512
    assert encoder_hidden_states.stride(-1) == 1
    assert encoder_hidden_states.stride(-2) == 4096
    assert decoder_hidden_states.stride(-1) == 1
    assert decoder_hidden_states.stride(-2) == 4096
    assert logit_weights.stride(-1) == 1
    assert logit_weights.stride(-2) == 512
    assert decoder_extended_attention_mask is None or decoder_extended_attention_mask.stride(-1) == 1
    assert decoder_extended_attention_mask is None or decoder_extended_attention_mask.stride(-2) == 4096
    assert decoder_extended_attention_mask is None or (
        decoder_extended_attention_mask.size(0) == batch_size and decoder_extended_attention_mask.size(1) == seqlen
    )
    assert decoder_hidden_states.size() == (
        batch_size,
        seqlen,
        512,
    )  # TODO: The condition should be derived from model config
    assert cell_state.stride(-1) == 1
    assert cell_state.stride(-2) == 4096
    assert cell_state.size() == (
        batch_size,
        seqlen,
        512,
    )  # TODO: The condition should be derived from model config

    assert prompt_lengths.stride(-1) == 1
    assert (
        init_state.stride(-1) == 1
    )  # Bug, the stride here does not follow the normal rule. The normal stride should be 512 for the last dim
    assert (
        init_state.stride(-2) == 512
    )  # Bug, the stride here does not follow the normal rule. The normal stride should be 1 for the second last dim

    stride_hidden_state_batch = hidden_state.stride(0)
    stride_hidden_state_head = hidden_state.stride(1)
    stride_hidden_state_head_step = hidden_state.stride(2)

    stride_encoder_hidden_states_batch = encoder_hidden_states.stride(0)
    stride_encoder_hidden_states_head = encoder_hidden_states.stride(1)
    stride_encoder_hidden_states_head_step = encoder_hidden_states.stride(2)

    stride_dec_hidden_states_batch = decoder_hidden_states.stride(0)
    stride_dec_hidden_states_head = decoder_hidden_states.stride(1)
    stride_dec_hidden_states_head_step = decoder_hidden_states.stride(2)

    assert stride_hidden_state_batch == stride_encoder_hidden_states_batch
    assert stride_hidden_state_batch == stride_dec_hidden_states_batch
    assert stride_hidden_state_head == stride_dec_hidden_states_head
    assert stride_hidden_state_head_step == 1
    assert stride_encoder_hidden_states_head_step == 1
    assert stride_dec_hidden_states_head_step == 1
    hidden_state_ptr = hidden_state.data_ptr()
    cell_state_ptr = cell_state.data_ptr()

    prompt_lengths_ptr = prompt_lengths.data_ptr()
    output_ptr = output.data_ptr()
    logit_weight_ptr = logit_weights.data_ptr()
    logit_bias_ptr = logit_bias.data_ptr()
    init_state_ptr = init_state.data_ptr()
    seqlen = output.size(1)
    extended_attention_mask_ptr = (
        decoder_extended_attention_mask.data_ptr() if decoder_extended_attention_mask is not None else 0
    )
    # this vllm setting assumes layernorm position, ffn, and qkv are part of the decoder sibling module.

    if init_state.dtype == torch.float16:
        dtype = "fp16"
    elif init_state.dtype == torch.float32:
        dtype = "fp32"
    else:
        raise Exception("Unknown dtype " + str(init_state.dtype))

    kwargs = {
        "opacity": args.opacity,
        "stream_count": 1,
        "small_chunk_size": 128 * 1024,
        "workspace_size": 16 * 1024 * 1024 * 1024,  # 16GB for the workspace
        "num_warps": 1,
        "coalesce": False,
        "preferred_block_size": 128,
    }
    assert args.block_1d <= 128
    ffn_intermediate_size = 4 * 1024 * 1024

    assert logit_weights.size(0) == batch_size, logit_weights.size()

    bwd_kernel_prompt_lengths = triton.jit(
        fwd(
            args.block_1d,
            -1,
            no_softmax=True,
            bias=True,
            pre_hook=pre_hook_cdiv,
            fp8_encode=init_state.dtype in [torch.float8_e4m3fn, torch.bfloat8],
        ),
        warm_cache=False,
        target="llm",
    )(partial(LLMBlock, ffn_intermediate_size=ffn_intermediate_size))

    ffn_intermediate_init = ffn_intermediate_init.view((-1, ffn_intermediate_size))
    bwd_kernel_prompt_lengths[(
        (batch_size,),
        (ceildiv(seqlen, args.block_1d),),
        (1,),
    )](
        output_ptr,
        init_state_ptr,
        prompt_lengths_ptr,
        hidden_state_ptr,
        cell_state_ptr,
        ffn_intermediate_init.data_ptr(),
        logit_weight_ptr,
        logit_bias_ptr,
        prompt_lengths_ptr
