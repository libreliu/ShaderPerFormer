import math
import logging
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional

from transformers.pytorch_utils import apply_chunking_to_forward

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_perfformer import PerfformerConfig
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput

import numpy as np
import warnings
def warn_once(message, category=UserWarning):
    if not hasattr(warn_once, "has_warned"):
        warn_once.has_warned = {}
    message_id = id(message)
    if message_id not in warn_once.has_warned:
        warnings.warn(message, category)
        warn_once.has_warned[message_id] = True

logger = logging.getLogger(__name__)

class MAPELoss(nn.Module):
    # https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/2
    __constants__ = ['reduction', 'eps']

    def __init__(self, epsilon=1e-4, reduction: str = 'mean') -> None:
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        assert(self.reduction in ['mean'])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lossVec = torch.abs((input - target) / torch.max(torch.ones_like(target) * self.epsilon, target))
        if self.reduction == 'mean':
            lossVec = torch.mean(lossVec)

        return lossVec

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))

@dataclass
class MultiTaskClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PerfformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PerfformerConfig
    base_model_prefix = "perfformer"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

class PerfformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute-learnable"
        )

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.position_embedding_type == "absolute-learnable":
            self.padding_idx = config.pad_token_id
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False
        )

        self.trace_label_embedding_type = config.trace_label_embedding_type
        self.hidden_size = config.hidden_size

        if self.trace_label_embedding_type == "binary-learnable":
            self.trace_label_binary_embedding_max_length = config.trace_label_binary_embedding_max_length
            self.trace_label_binary_embedding = nn.Embedding(
                self.trace_label_binary_embedding_max_length * 2,
                config.hidden_size
            )

    def _calculate_trace_label_embeddings_sine(self, trace_labels: 'torch.Tensor') -> 'torch.Tensor':
        raise NotImplementedError("TODO: implement sine encoding")

    def _calculate_input_id_embeddings_sine(self, input_ids: 'torch.Tensor') -> 'torch.Tensor':
        raise NotImplementedError("TODO: implement sine encoding")

    def _calculate_trace_label_embeddings_binary_learnable(self, trace_labels: 'torch.Tensor') -> 'torch.Tensor':
        d_model = self.hidden_size

        # this assumes all input number < 2**bits_max-1
        bits_max = 64

        bsz, seq_len = trace_labels.size()
        trace_labels = trace_labels.to(dtype=torch.int64, device=trace_labels.device)
        ones = torch.ones((bits_max,), dtype=torch.int64, device=trace_labels.device)

        # (bits_max, )
        masks = torch.bitwise_left_shift(ones, torch.arange(bits_max, device=trace_labels.device))

        # shape (bsz, seq_len, bits_max)
        # bits_max dim is from LSB first
        binary_form = (torch.bitwise_and(trace_labels.view(bsz, seq_len, 1), masks.view(1, 1, bits_max)) > 0).int()

        # check if some trace_label have gone beyond the limit
        d_emb_max_len = self.trace_label_binary_embedding_max_length

        # (bsz, seq_len)
        is_beyond_limit_mask = (
            torch.sum(binary_form[:, :, d_emb_max_len:], dim=2) > 0
        )

        # shape (bsz, seq_len, d_emb_max_len)
        binary_form_masked = torch.where(
            # (bsz, seq_len, 1)
            is_beyond_limit_mask.unsqueeze(-1),
            # (1, 1, 1)
            torch.ones((1, 1, 1), dtype=torch.int64, device=trace_labels.device),
            # (bsz, seq_len, d_emb_max_len)
            binary_form[:, :, :d_emb_max_len]
        )

        # Manually unwraps the first loop
        assert(self.trace_label_binary_embedding_max_length >= 1)
        # (bsz, seq_len, d_model = d_model)
        trace_embeds = torch.where(
            # (bsz, seq_len) -> (bsz, seq_len, 1)
            (binary_form_masked[:, :, 0] == 1).view(bsz, seq_len, 1),
            # The embedding of 1, (d_model,) -> (1, 1, d_model)
            self.trace_label_binary_embedding(torch.as_tensor(1, device=binary_form_masked.device)).view(1, 1, d_model),
            # The embedding of 0, (d_model,) -> (1, 1, d_model)
            self.trace_label_binary_embedding(torch.as_tensor(1, device=binary_form_masked.device)).view(1, 1, d_model)
        )

        for idx in range(1, d_emb_max_len):
            trace_embeds += (
                torch.where(
                    (binary_form_masked[:, :, idx] == 1).view(bsz, seq_len, 1),
                    self.trace_label_binary_embedding(torch.as_tensor(2 * idx + 1, device=binary_form_masked.device)).view(1, 1, d_model),
                    self.trace_label_binary_embedding(torch.as_tensor(2 * idx, device=binary_form_masked.device)).view(1, 1, d_model)
                )
            )

        return trace_embeds

    def forward(
        self,
        input_ids: 'torch.Tensor',
        token_type_ids=None,
        position_ids=None,
        # shape (bsz, seqlen)
        trace_labels=None,
        # shape (bsz, seqlen, d_model), and mean=0 std=1 should be better
        trace_embeds=None
    ):
        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            # This actually assumes a "left-padded" token to be inputed
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, 0)

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.input_ids.device)

        # check sanity
        if self.trace_label_embedding_type == "none":
            assert(trace_labels is None and trace_embeds is None)
        elif self.trace_label_embedding_type == "sine":
            assert(trace_labels is not None and trace_embeds is None)
            trace_embeds = self._calculate_trace_label_embeddings_sine(trace_labels)
        elif self.trace_label_embedding_type == "input":
            assert(trace_embeds is not None)
        elif self.trace_label_embedding_type == "binary-learnable":
            assert(trace_labels is not None and trace_embeds is None)
            trace_embeds = self._calculate_trace_label_embeddings_binary_learnable(trace_labels)

        embeddings = self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)

        if trace_embeds is not None:
            embeddings += trace_embeds

        position_embeddings = None
        if self.position_embedding_type == "absolute-learnable":
            position_embeddings = self.position_embeddings(position_ids)
        elif self.position_embedding_type == "absolute-sine":
            position_embeddings = self._calculate_input_id_embeddings_sine(input_ids)
        elif self.position_embedding_type == "rope":
            # do nothing
            pass
        else:
            raise NotImplementedError(f"Unknown position embedding type {self.position_embedding_type}")

        if position_embeddings is not None:
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class PerfformerSelfAttention(nn.Module):
    def __init__(self, config: 'PerfformerConfig', position_embedding_type=None):
        super().__init__()

        assert(config.attention_type in (
            "vanilla",
            "torch-memeff-nomask", "torch-flash-nomask",
            "xformers-memeff", "xformers-memeff-nomask"
        ))

        if config.attention_type.startswith("torch"):
            pytorch_major_version = int(torch.__version__.split('.')[0])
            if pytorch_major_version < 2:
                raise NotImplementedError("Fast SPDA operator requires PyTorch version >= 2")
            
            self.torch_backend_map = {
                'torch-math': {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
                'torch-flash-nomask': {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
                'torch-memeff-nomask': {
                    "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
            }

        elif config.attention_type.startswith("xformers"):
            try:
                import xformers.ops
            except ImportError:
                raise NotImplementedError("Need xformers to be installed. Please install with pip install -U xformers")

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.attention_type = config.attention_type
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        if config.attention_type == "vanilla":
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout_prob = config.attention_probs_dropout_prob


        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute-learnable"
        )

        # Got other stuff killed
        assert(self.position_embedding_type in (
            'absolute-learnable', 'rope', 'absolute-sine'
        ))

    def transpose_for_scores(
            self,
            x: torch.Tensor    # shape: (batch_size, seqlen, d_model)
        ) -> torch.Tensor:
        # (bsz, seqlen, n_heads, d_head)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # (bsz, n_heads, seq_len, d_head)
        return x.permute(0, 2, 1, 3)

    # Copied from RoformerSelfAttention
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    def forward(
        self,
        # hidden_states: (bsz, seq_len, d_model)
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        sinusoidal_pos: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:

        # (bsz, num_heads, seq_len, head_size)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.position_embedding_type == "rope":
            assert(sinusoidal_pos is not None)
            query_layer, key_layer = self.apply_rotary_position_embeddings(
                sinusoidal_pos, query_layer, key_layer
            )

            # cast to fp16 when using mixamp
            if query_layer.dtype != value_layer.dtype:
                query_layer = query_layer.to(value_layer.dtype)

            if key_layer.dtype != value_layer.dtype:
                key_layer = key_layer.to(value_layer.dtype)

        # functional attention does not support this
        if self.attention_type == "torch-memeff-nomask" or self.attention_type == "torch-flash-nomask":
            # q, k, v: (bsz, num_heads, seq_len, head_dim)
            # returns: 
            if attention_mask is not None:
                warn_once("The attention mask is not None")

            with torch.backends.cuda.sdp_kernel(**self.torch_backend_map[self.attention_type]):
                context_layer = torch.nn.functional.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer,
                    attn_mask=None,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=False
                )

        elif self.attention_type == "xformers-memeff-nomask":
            if attention_mask is not None:
                warn_once("The attention mask is not None")

            # q, k, v: (bsz, seq_len, num_heads, head_dim)
            import xformers.ops
            context_layer = xformers.ops.memory_efficient_attention(
                query_layer.permute(0, 2, 1, 3),
                key_layer.permute(0, 2, 1, 3),
                value_layer.permute(0, 2, 1, 3),
                attn_bias=None,
                p=self.dropout_prob if self.training else 0.0,
                scale=None
            ).permute(0, 2, 1, 3)
        elif self.attention_type == "xformers-memeff":

            # q, k, v: (bsz, seq_len, num_heads, head_size)
            # output are of the same layout, so permute back
            import xformers.ops
            context_layer = xformers.ops.memory_efficient_attention(
                query_layer.permute(0, 2, 1, 3),
                key_layer.permute(0, 2, 1, 3),
                value_layer.permute(0, 2, 1, 3),
                # (bsz, num_heads, seq_len, seq_len)
                attn_bias=attention_mask.expand(query_layer.size(0), query_layer.size(1), query_layer.size(2), query_layer.size(2)).to(dtype=query_layer.dtype),
                p=self.dropout_prob if self.training else 0.0,
                scale=None
            ).permute(0, 2, 1, 3)
        elif self.attention_type == "vanilla":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
                attention_scores = attention_scores + attention_mask
            
             # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
        else:
            raise NotImplementedError(f"Unknown attention type {self.attention_type}")

        # (bsz, num_heads, seq_len, head_size) -> (bsz, seqlen, n_heads, d_head)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # shape (bsz, seqlen, d_model)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class PerfformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class PerfformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# Copied from transformers.models.bert.modeling_bert.BertOutput
class PerfformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class PerfformerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = PerfformerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = PerfformerSelfOutput(config)

    # hidden_states is of (bsz, seqlen, d_model), and the output is the same
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        sinusoidal_pos: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:
        self_output = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos
        )

        attention_output = self.output(self_output, hidden_states)
        return attention_output

class PerfformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerfformerAttention(config)
        self.intermediate = PerfformerIntermediate(config)
        self.output = PerfformerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        sinusoidal_pos: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos
        )

        # https://huggingface.co/docs/transformers/main/glossary#feed-forward-chunking
        # mathematically equivalent to calling self.feed_forward_chunk directly
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

# Copied from Roformer
class PerfformerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

class PerfformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [PerfformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute-learnable"
        )

        # Got other stuff killed
        assert(self.position_embedding_type in (
            'absolute-learnable', 'rope', 'absolute-sine'
        ))

        if self.position_embedding_type == 'rope':
            self.embed_positions = PerfformerSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.hidden_size // config.num_attention_heads
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        # Output the hidden state (input, after_1st_layer, after_2nd_layer, ..., after_last_layer)
        output_hidden_states: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            sinusoidal_pos = None

            if self.position_embedding_type == 'rope':
                # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
                sinusoidal_pos = self.embed_positions(
                    hidden_states.shape[:-1],
                    # past_key_values_length
                    0
                )[None, None, :, :]

            if self.gradient_checkpointing and self.training:
                layer_output = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos
                )
            else:
                layer_output = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos
                )

            hidden_states = layer_output

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states
        )

class PerfformerModel(PerfformerPreTrainedModel):

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = PerfformerEmbeddings(config)
        self.encoder = PerfformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError()

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        trace_labels: Optional[torch.Tensor] = None,
        trace_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            trace_labels=trace_labels,
            trace_embeds=trace_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs.last_hidden_state

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states
        )

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class PerfformerLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = ACT2FN['gelu'](x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # NOTE: roberta-base state_dicts have only self.bias available
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

class PerfformerForMaskedLM(PerfformerPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.perfformer = PerfformerModel(config)
        self.lm_head = PerfformerLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    # NOTE: tie_word_embeddings defaults to True (see PretrainedConfig class)
    # and when it enables, PretrainedModel.tie_weights() will tie the things together
    # =====
    # The code is as follows:
    # if getattr(self.config, "tie_word_embeddings", True):
    #        output_embeddings = self.get_output_embeddings()
    #        if output_embeddings is not None:
    #            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.perfformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states
        )


class PerfformerRegressionHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        regression_hidden_dim = (
            config.regression_hidden_dim if config.regression_hidden_dim is not None else config.hidden_size
        )
        self.mlp = nn.Sequential(*[
            nn.Linear(config.hidden_size, regression_hidden_dim),
            nn.Tanh()
        ])
        regression_dropout = (
            config.regression_dropout if config.regression_dropout is not None else config.hidden_dropout_prob
        )

        self.regression_head_configuration = getattr(
            config, "regression_head_configuration", "bos-reduction"
        )

        assert(self.regression_head_configuration in ("bos-reduction", "seq-sum-reduction"))
        logger.info(f"Using regression head configuration {self.regression_head_configuration}")

        if regression_dropout != 0.0:
            self.dropout = nn.Dropout(regression_dropout)
        else:
            self.dropout = None

        self.out_proj = nn.Linear(regression_hidden_dim, 1)

    def forward(self, features, **kwargs):
        # bsz, slen, dmodel
        if self.regression_head_configuration == "seq-sum-reduction":
            x = features
            if self.dropout is not None:
                x = self.dropout(x)

            x = self.mlp(x)
            if self.dropout is not None:
                x = self.dropout(x)

            # (bsz, seq_len, 1) -> (bsz, 1)
            x = torch.sum(self.out_proj(x), dim=1)
        elif self.regression_head_configuration == "bos-reduction":
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
            if self.dropout is not None:
                x = self.dropout(x)

            x = self.mlp(x)
            if self.dropout is not None:
                x = self.dropout(x)

            x = self.out_proj(x)
        else:
            raise NotImplementedError(f"Unknown regression head configuration {self.regression_head_configuration}")
        return x

class PerfformerForRegression(PerfformerPreTrainedModel):
    """
    To use:
    - config.problem_type: one of ['regression-xxxx']
    - config.regression_dropout: defaults to config.hidden_dropout_prob
    - config.regression_head_dim: defaults to config.hidden_size
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.perfformer = PerfformerModel(config)
        self.regression_head = PerfformerRegressionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        trace_labels: Optional[torch.FloatTensor] = None,
        trace_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None
    ) -> SequenceClassifierOutput:

        outputs = self.perfformer(
            input_ids,
            trace_labels=trace_labels,
            trace_embeds=trace_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )

        # shape (bsz, seq_len, d_model)
        sequence_output = outputs["last_hidden_state"]

        bsz, seqlen, d_model = sequence_output.size()

        # shape (bsz, 1)
        logits = self.regression_head(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # shape (bsz, )
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression-mse":
                loss_fct = MSELoss()
            elif self.config.problem_type == "regression-mape":
                loss_fct = MAPELoss()
            elif self.config.problem_type == "regression-msle":
                loss_fct = MSLELoss()
            elif self.config.problem_type == "regression-inv-sample-weighted-mse":
                pass
            else:
                raise NotImplementedError(f"Unknown problem type {self.config.problem_type}")

            if self.config.problem_type == "regression-inv-sample-weighted-mse":
                loss = (labels * (logits.view((bsz,)) - labels) ** 2).mean()
            else:
                loss = loss_fct(logits.view((bsz,)), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )

class PerfformerForMultiArchLearning(PerfformerPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.mal_num_lm_heads = config.mal_num_lm_heads

        self.perfformer = PerfformerModel(config)
        self.regression_heads = nn.ModuleList(
            [PerfformerRegressionHead(config) for _ in range(0, self.mal_num_lm_heads)]
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # This serves as the target - used in Regression task
        # (bsz,)
        labels: Optional[torch.LongTensor] = None,
        # (bsz,) - integer specifying which head to use
        label_dest_heads: Optional[Union[torch.LongTensor, List[int]]] = None,
        trace_labels: Optional[torch.FloatTensor] = None,
        trace_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None
    ) -> SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.perfformer(
            input_ids,
            trace_labels=trace_labels,
            trace_embeds=trace_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )
        # shape (bsz, seq_len, d_model)
        sequence_output = outputs.last_hidden_state

        bsz, seqlen, d_model = sequence_output.size()

        # shape (bsz, 1)
        assert(label_dest_heads is not None)
        logits = torch.zeros((bsz,1), device=sequence_output.device)
        for idx, destHeadIdx in enumerate(label_dest_heads):
            logits[idx, 0] = self.regression_heads[int(destHeadIdx)](sequence_output[idx, :, :].view(1, seqlen, d_model))

        regression_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # shape (bsz, )
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression-mse":
                loss_fct = MSELoss()
            elif self.config.problem_type == "regression-mape":
                loss_fct = MAPELoss()
            elif self.config.problem_type == "regression-msle":
                loss_fct = MSLELoss()
            else:
                raise NotImplementedError(f"Unknown problem type {self.config.problem_type}")

            regression_loss = loss_fct(logits.view((bsz,)), labels)

        return SequenceClassifierOutput(
            loss=regression_loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )

class PerfformerForMultiTaskLearning(PerfformerPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.mtl_mlm_loss_weight = config.mtl_mlm_loss_weight
        self.mtl_regression_loss_weight = config.mtl_regression_loss_weight

        self.perfformer = PerfformerModel(config)
        self.regression_head = PerfformerRegressionHead(config)
        self.lm_head = PerfformerLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # This serves as the target - used in Regression task
        labels: Optional[torch.LongTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        trace_labels: Optional[torch.FloatTensor] = None,
        trace_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.perfformer(
            input_ids,
            trace_labels=trace_labels,
            trace_embeds=trace_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )
        # shape (bsz, seq_len, d_model)
        sequence_output = outputs.last_hidden_state

        bsz, seqlen, d_model = sequence_output.size()

        # lm_head output
        if self.mtl_mlm_loss_weight > 0:
            lm_prediction_scores = self.lm_head(sequence_output)

            masked_lm_loss = None
            if mlm_labels is not None:
                # move labels to correct device to enable model parallelism
                mlm_labels = mlm_labels.to(lm_prediction_scores.device)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if self.mtl_regression_loss_weight > 0:
            # shape (bsz, 1)
            logits = self.regression_head(sequence_output)

            regression_loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                # shape (bsz, )
                labels = labels.to(logits.device)

                if self.config.problem_type == "regression-mse":
                    loss_fct = MSELoss()
                elif self.config.problem_type == "regression-mape":
                    loss_fct = MAPELoss()
                elif self.config.problem_type == "regression-msle":
                    loss_fct = MSLELoss()
                else:
                    raise NotImplementedError(f"Unknown problem type {self.config.problem_type}")

            regression_loss = loss_fct(logits.view((bsz,)), labels)

        loss = torch.zeros((1,), device=sequence_output.device)
        if self.mtl_mlm_loss_weight > 0 and masked_lm_loss is not None:
            loss += self.mtl_mlm_loss_weight * masked_lm_loss

        if self.mtl_regression_loss_weight > 0 and regression_loss is not None:
            loss += self.mtl_regression_loss_weight * regression_loss

        return MultiTaskClassifierOutput(
            loss=loss[0],
            # Need to ignore on inference, or it will be combined into logits
            mlm_logits=lm_prediction_scores,
            logits=logits,
            hidden_states=outputs.hidden_states
        )


class PerfformerForMoCo(PerfformerPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.symmetric = config.moco_symmetric
        self.shuffle_batch = config.moco_shuffle_batch
        self.K = config.moco_K
        self.m = config.moco_m
        self.T = config.moco_T
        self.use_bn = config.moco_use_bn

        self.encoder_q = PerfformerModel(config)
        self.encoder_k = PerfformerModel(config)

        # initialize encoder_k as a copy of encoder_q
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.encoder_k.requires_grad_(False)

        self.register_buffer("queue", torch.randn(config.hidden_size, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert(self.K % batch_size == 0)
        assert(self.K == self.queue.size(1))

        if ptr + batch_size >= self.K:
            trailings = self.K - ptr
            beginnings = ptr + batch_size - self.K

            transposed = keys.t()
            self.queue[:, ptr:self.K] = transposed[:, 0:trailings]
            self.queue[:, 0:beginnings] = transposed[:, trailings:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.t()

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, feature: Dict[str, torch.Tensor]):
        # random shuffle index
        idx_shuffle = torch.randperm(feature['input_ids'].size(0), device=feature['input_ids'].device)
        idx_unshuffle = torch.argsort(idx_shuffle)

        for k, v in feature.items():
            feature[k] = v[idx_shuffle]

        return feature, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]
    
    def contrastive_loss(self, feature_q: dict, feature_k: dict):
        q = self.encoder_q(**feature_q)
        q = q["last_hidden_state"]

        if self.use_bn:
            q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            if self.shuffle_batch:
                shuffle_k, idx_unshuffle = self._batch_shuffle_single_gpu(feature_k)

                k = self.encoder_k(**shuffle_k)
                k = k["last_hidden_state"]

                if self.use_bn:
                    k = nn.functional.normalize(k, dim=1)

                k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
            else:
                k = self.encoder_k(**feature_k)
                k = k["last_hidden_state"]

                if self.use_bn:
                    k = nn.functional.normalize(k, dim=1)

        q = q[:, 0, :]
        k = k[:, 0, :]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        bsz = logits.size(0)
        labels = torch.zeros(bsz, dtype=torch.long, device=logits.device)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss, q, k

    def forward(self, feature_q: dict, feature_k=None, labels=None):
    
        if feature_k is None:
            assert(not self.training)

            q = self.encoder_q(**feature_q)

            return {
                "logits": q["last_hidden_state"][:, 0, :],
                "loss": torch.zeros(q["last_hidden_state"].size(0), device=q["last_hidden_state"].device)
            }
        else:
            assert(self.training)

        if self.training:
            with torch.no_grad():
                self._momentum_update_key_encoder()

        if self.symmetric:
            loss_12, q1, k1 = self.contrastive_loss(feature_q, feature_k)
            loss_21, q2, k2 = self.contrastive_loss(feature_k, feature_q)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:
            loss, q, k = self.contrastive_loss(feature_q, feature_k)

        self._dequeue_and_enqueue(k)

        # return loss
        return SequenceClassifierOutput(
            loss=loss
        )