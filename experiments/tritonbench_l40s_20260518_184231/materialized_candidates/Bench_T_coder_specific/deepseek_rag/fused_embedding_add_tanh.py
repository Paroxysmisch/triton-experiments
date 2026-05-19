)
        BLOCK_SIZE = 256
        N = weight.shape[1]
        out = torch.empty_like(indices).cuda()

        grid = lambda M: (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        embedding_kernel[grid(M), BLOCK_SIZE](
            out_ptr=out.data_ptr(),
            in_ptr=indices.data_ptr(),
            weight_ptr=weight.data_ptr(),
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if scale_grad_by_freq:
            indices_freq = torch.zeros(weight.shape[0]).cuda().int()
            INDICE_BLOCK_SIZE = 256
            grid = lambda M: (M + INDICE_BLOCK_SIZE - 1) // INDICE_BLOCK_SIZE
            indice_freq_kernel[grid(M), INDICE_BLOCK_SIZE](
                indices_freq=indices_freq.data_ptr(),
                indices=indices.data_ptr(),
                elem_cnt=M,
                INDICE_BLOCK_SIZE=INDICE_BLOCK_SIZE,
            )
            grad_scale = torch.ones_like(indices_freq)
            GRAD_SCALE_BLOCK_SIZE = 256
            grid = lambda M: (M + GRAD_SCALE_BLOCK_SIZE - 1) // GRAD_SCALE_BLOCK_SIZE
            embedding_grad_scale_kernel[grid(M), GRAD_SCALE_BLOCK_SIZE](
                grad_out=out.data_ptr(),
                indice_freq=indices_freq.data_ptr(),
                n_rows=weight.shape[0],
                N=N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.scale_grad_by_freq = grad_scale

        ctx.save_for_backward(indices, weight, out)
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return out

    @staticmethod
    def backward(ctx, grad_output):
        indices, weight, out = ctx.saved_tensors
        padding_idx = ctx.padding_idx
        scale_grad_by_freq = ctx.scale_grad_by_freq
        sparse = ctx.sparse
        M = math.prod(indices.shape)
        N = weight.shape[1]
        BLOCK_SIZE = 256

        grad_input = torch.empty_like(indices)
        embedding_backward_kernel[grid(M), BLOCK_SIZE](
            grad_in=grad_input.data_ptr(),
            grad_out=grad_output.data_ptr(),
            indices=indices.data_ptr(),
            padding_idx=padding_idx,
            HAS_PADDING_IDX=padding_idx >= 0,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if scale_grad_by_freq:
            grad_input = grad_input * ctx.scale_grad_by_freq

        grad_weight = torch.empty_like(weight)
        tl.launch_kernel(
            embedding_backward_kernel,
            grid(M),
            BLOCK_SIZE,
            args=[
                grad_output.data_ptr(),
                grad_input.data_ptr(),
                indices.data_ptr(),
                padding_idx,
                padding_idx >= 0,
                N,
                BLOCK_SIZE,
            ],
            stream=torch.cuda.current_stream(),
        )

        return grad_weight, grad_input, None, None, None, None


def fused_embedding_add_tanh(
    input_indices, weight, other, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None
):
    if padding_idx is None:
        padding_idx = -1
    if out is None:
        out = torch.empty_like(input_indices)
    out = Embedding.apply(weight, input_indices, padding_idx, scale_grad_by_freq, sparse) + other
    out = torch.tanh(out)
    return out
|end-of-text|
"""

#src/models/model_factory.py
from models.transformer import Transformer
from models.seq2seq import Seq2Seq
from models.language_model import LanguageModel
from models.gpt2 import GPT2
from models.bert import BERT
from models.triton_bert import TritonBERT
from models.t5 import T5
from models.triton_t5 import TritonT5


def create_model(model_name: str, device: str):
    if model_name == "transformer":
        return Transformer(device)
    elif model_name == "seq2seq":
        return Seq2Seq(device)
    elif model_name == "language_model":
        return LanguageModel(device)
    elif model_name == "gpt2":
        return GPT2(device)
    elif model_name == "bert":
        return BERT(device)
    elif model_name == "triton_bert":
        return TritonBERT(device)
    elif model_name == "t5":
        return T5(device)
    elif model_name == "triton_t5":
        return TritonT5(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

#src/preprocessing/preprocessing.py
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torchtext.data import FunctionFilter, Field, BucketIterator
from typing import Tuple


def yield_tokens(data_iter: Multi30k) -> Tuple[str, str]:
    for data in data_iter:
        src, tgt = data.src, data.tgt
        yield src
        yield tgt


def preprocess_data(batch_size: int, device: str) -> Tuple[Field, BucketIterator]:
    token_transform = get_tokenizer('spacy', language='de')
    vocab_transform = build_vocab_from_iterator(yield_tokens(Multi30k(split='train', language_pair=('de', 'en'))),
                                                min_freq=1,
                                                specials=["<unk>", "<sos>", "<eos>"])
    vocab_transform.set_default_index(0)

    SRC = Field(tokenize=token_transform,
                init_token="<sos>",
                eos_token="<eos>",
                lower=True,
                tokenizer_language='de',
                batch_first=True,
                include_lengths=True)

    TRG = Field(tokenize=token_transform,
                init_token="<sos>",
                eos_token="<eos>",
                lower=True,
                tokenizer_language='en',
                batch_first=True)

    train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))
    train_data = train_data.filter(lambda x: len(x.src) <= 200 and len(x.tgt) <= 100)
    train_data, valid_data = train_data.split(random_state=torch.manual_seed(1234))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)
