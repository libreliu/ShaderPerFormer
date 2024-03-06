import math
from typing import Tuple, Union

import torch
import torch.nn as nn

# https://datascience.stackexchange.com/questions/65067/proper-masking-in-the-transformer-model
# https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

class PerformanceNet(nn.Module):
    """A transformer encoder, with MLP head"""

    def __init__(self, ntoken: int, d_model: int, d_mlp: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, output_dim: int = 1):

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, output_dim)
        )

        self.init_weights()
        
        n_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder[0].bias.data.zero_()
        self.decoder[2].bias.data.zero_()
        self.decoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[2].weight.data.uniform_(-initrange, initrange)

    # https://discuss.pytorch.org/t/mse-ignore-index/43934
    def mse_loss(self, input, target, ignored_index, reduction):
        assert(False)
        mask = target == ignored_index
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out

    def forward(
        self,
        src: torch.Tensor,
        targets = None,
        ignored_index = None,
        src_mask = None,
        src_key_padding_mask = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            targets: Tensor, shape ``[batch_size, output_dim]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``, the addictive mask
            src_key_padding_mask: Tensor, shape ``[batch_size, seq_len]``.  If a BoolTensor is provided,
            the positions with the value of True will be ignored while the position with the value of 
            False will be unchanged.

        Returns:
            output Tensor of shape ``[batch_size, output_dim]``, or ``[1]`` (mse loss) when targets given
        """
        # print(src.shape)
        # src = src.transpose(0, 1)
        # print(src.shape)
        # (bsz, seqlen) -> (bsz, seqlen, d_model)
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # print(src.shape)
        src = self.pos_encoder(src)
        # print(src.shape)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)

        # (seq_len, batch_size, d_model) -> (batch_size, d_model)
        # print(output.shape)
        output = output[0, :, :]
        # print(output.shape)

        # (batch_size, output_dim)
        output = self.decoder(output)

        if targets is not None:
            if ignored_index is None:
                loss = torch.nn.functional.mse_loss(output, targets)
            else:
                loss = self.mse_loss(output, targets, ignored_index, "mean")

            return output, loss
        else:
            return output

    def configure_optimizers(self, weight_decay: float = 0.1, learning_rate: float=3e-4, betas: Tuple[float, float]=(0.9, 0.95)):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        print(f"weight_decay={weight_decay} learning_rate={learning_rate} betas={betas}")

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('in_proj_weight'):
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

