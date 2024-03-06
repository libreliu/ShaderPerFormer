from typing import List
from transformers.tokenization_utils_base import (
    BatchEncoding
)
import torch
import functools

def do_onehot_base2(trace_labels_raw: torch.Tensor, d_embed: int, shifted=False) -> 'torch.Tensor':
    bsz, seq_len = trace_labels_raw.size()
    trace_labels = trace_labels_raw.to(dtype=torch.int64, device=trace_labels_raw.device)
    ones = torch.ones((d_embed,), dtype=torch.int64, device=trace_labels.device)

    # (d_embed, )
    masks = torch.bitwise_left_shift(ones, torch.arange(d_embed, device=trace_labels.device))
    trace_embeds = (torch.bitwise_and(trace_labels.view(bsz, seq_len, 1), masks.view(1, 1, d_embed)) > 0).float()
    if shifted:
        trace_embeds -= 0.5

    return trace_embeds

DISPATCH_TABLE = {
    'onehot-base2': do_onehot_base2,
    'onehot-base2-shifted': functools.partial(do_onehot_base2, shifted=True)
}

def generate_trace_embedding(method: str, batch: BatchEncoding, d_embed: int) -> BatchEncoding:
    """Returns a (seq_len, d_embed) nested list"""
    if "trace_labels" not in batch:
        raise RuntimeError(f"trace_labels is not present in batch. Got {batch.keys()}")
    
    if not isinstance(batch["trace_labels"], torch.Tensor):
        raise RuntimeError(f"Expected torch tensor as input")

    bsz, seqlen = batch["trace_labels"].size()
    trace_embeds = DISPATCH_TABLE[method](batch["trace_labels"], d_embed)

    batch["trace_embeds"] = trace_embeds
    return batch

