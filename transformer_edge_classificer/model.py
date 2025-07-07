# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter
from typing import Optional, Tuple, List
import numpy as np
import math
import config as model_config


# CustomMultiheadAttention
class CustomMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim: raise ValueError(f"embed_dim error")
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None: nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None: nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None: nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.batch_first:
            bsz, tgt_len, embed_dim = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
            query,key,value = query.transpose(0,1),key.transpose(0,1),value.transpose(0,1)
        if embed_dim != self.embed_dim: raise ValueError(f"Query embed_dim error")
        
        num_heads, head_dim = self.num_heads, self.head_dim
        q,k,v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                if attn_mask.shape[1] == num_heads and tgt_len == attn_mask.shape[2]: attn_mask_expanded = attn_mask
                elif attn_mask.shape[1] == tgt_len: attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                else: raise ValueError(f"attn_mask dim 3 shape error: {attn_mask.shape}, H={num_heads}, T={tgt_len}, S={src_len}")
            elif attn_mask.dim() == 2: attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1, -1)
            else: raise ValueError(f"attn_mask dim error: {attn_mask.shape}")
            
            # The JIT compiler can correctly trace operations on a ByteTensor (0s and 1s) for masking.
            # Explicitly converting to a boolean with .bool() can sometimes break the trace graph,
            # especially in older PyTorch versions. Using the ByteTensor directly is more robust for JIT.
            attn_scores = attn_scores.masked_fill(attn_mask_expanded, float('-inf'))
        
        attn_probs = F.softmax(attn_scores + 1e-9, dim=-1)
        attn_probs = attn_probs.masked_fill(torch.isinf(attn_scores), 0.0)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        if not self.batch_first: attn_output = attn_output.transpose(0, 1)
        return attn_output, None


class CustomTransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout_attn = Dropout(dropout)
        self.dropout_ff = Dropout(dropout)
        self.alpha_attn = Parameter(torch.tensor(0.0))
        self.alpha_ff = Parameter(torch.tensor(0.0))
        if isinstance(activation, str): self.activation = F.relu if activation == "relu" else F.gelu if activation == "gelu" else None
        else: self.activation = activation
        if self.activation is None: raise RuntimeError(f"activation error")

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        if self.norm_first:
            attn_block_output = self._sa_block(self.norm1(x), attn_mask, src_key_padding_mask)
            x = x + self.alpha_attn * self.dropout_attn(attn_block_output)
            ff_block_output = self._ff_block(self.norm2(x))
            x = x + self.alpha_ff * self.dropout_ff(ff_block_output)
        else:
            attn_block_output = self._sa_block(x, attn_mask, src_key_padding_mask)
            x = self.norm1(x + self.alpha_attn * self.dropout_attn(attn_block_output))
            ff_block_output = self._ff_block(x)
            x = self.norm2(x + self.alpha_ff * self.dropout_ff(ff_block_output))
        return x

    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return attn_output

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CustomTransformerEncoder(Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer_template, num_layers, norm=None):
        super().__init__()
        compiled_layers = []
        for _ in range(num_layers):
            layer = CustomTransformerEncoderLayer(
                d_model=encoder_layer_template.self_attn.embed_dim,
                nhead=encoder_layer_template.self_attn.num_heads,
                dim_feedforward=encoder_layer_template.linear1.out_features,
                dropout=encoder_layer_template.dropout.p,
                activation=encoder_layer_template.activation,
                layer_norm_eps=encoder_layer_template.norm1.eps,
                batch_first=encoder_layer_template.batch_first,
                norm_first=encoder_layer_template.norm_first
            )
            compiled_layers.append(torch.jit.script(layer))
        self.layers = ModuleList(compiled_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, attn_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEdgeClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes,
                 dropout=0.1, norm_first=False,
                 k_nearest: Optional[int] = None,
                 knn_distance_threshold: Optional[float] = None,
                 mlp_head_dims: Optional[List[int]] = None):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.k_nearest = k_nearest
        self.knn_distance_threshold = knn_distance_threshold
        self.distance_metric = "segment"

        if d_model % nhead != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_embed = nn.Linear(input_dim, d_model)

        custom_encoder_layer_template = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=norm_first
        )
        encoder_norm = LayerNorm(d_model, eps=custom_encoder_layer_template.norm1.eps) if not norm_first else None
        self.transformer_encoder = CustomTransformerEncoder(
            encoder_layer_template=custom_encoder_layer_template,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )

        if not mlp_head_dims:
            self.output_layer = nn.Linear(d_model, num_classes)
            mlp_head_str = "Linear"
        else:
            layers = []
            in_dim = d_model
            for hidden_dim in mlp_head_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, num_classes))
            self.output_layer = nn.Sequential(*layers)
            mlp_head_str = f"MLP {d_model} -> {' -> '.join(map(str, mlp_head_dims))} -> {num_classes}"

        self.init_weights()

        print(f"TransformerEdgeClassifier Initialized (with JIT-compiled Encoder)")
        print(f"  Input Dim: {input_dim}")
        print(f"  Embedding Dim (d_model): {d_model}")
        print(f"  Num Heads (nhead): {nhead}")
        print(f"  Num Encoder Layers: {num_encoder_layers}")
        print(f"  FeedForward Dim (dim_ff): {dim_feedforward}")
        print(f"  Normalization: {'Pre-LN' if norm_first else 'Post-LN (ReZero)'}")
        attn_type_str = "Full (Dense)"
        if k_nearest is not None and k_nearest > 0:
            attn_type_str = f"K-Nearest Lines (k={k_nearest}, {self.distance_metric.capitalize()} Distance)"
            if self.knn_distance_threshold is not None:
                attn_type_str += f", Threshold: {self.knn_distance_threshold:.3f}"
        print(f"  Attention Type: {attn_type_str}")
        print(f"  Classification Head: {mlp_head_str}")
        print(f"----------------------------------------------------------")

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embed.bias.data.zero_()
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        for module in self.output_layer.modules():
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
                module.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, feat_dim = src.shape
        device = src.device
        eps = 1e-9

        attn_mask = None
        if self.k_nearest is not None and self.k_nearest > 0 and seq_len > 1:
            k_actual = min(self.k_nearest, seq_len)

            x1_idx, y1_idx = model_config.X1_IDX, model_config.Y1_IDX
            x2_idx, y2_idx = model_config.X2_IDX, model_config.Y2_IDX
            if not all(idx >= 0 and idx < feat_dim for idx in [x1_idx, y1_idx, x2_idx, y2_idx]):
                 raise ValueError(f"Endpoint indices out of bounds for input features with shape {src.shape}")

            p1 = src[:, :, [x1_idx, y1_idx]]
            q1 = src[:, :, [x2_idx, y2_idx]]
            u = q1 - p1
            p1_i = p1.unsqueeze(2)
            u_i = u.unsqueeze(2)
            p2_j = p1.unsqueeze(1)
            v_j = u.unsqueeze(1)
            
            a = torch.sum(u_i * u_i, dim=-1)
            b = torch.sum(u_i * v_j, dim=-1)
            c = torch.sum(v_j * v_j, dim=-1)
            w0 = p1_i - p2_j
            
            d_dot = torch.sum(u_i * w0, dim=-1)
            e_dot = torch.sum(v_j * w0, dim=-1)

            denom = (a + eps) * c.expand(-1, seq_len, -1) - b * b
            
            tc_inf = (b * e_dot - c.expand(-1, seq_len, -1) * d_dot) / (denom + eps)
            sc_inf = (a.expand(-1, -1, seq_len) * e_dot - b * d_dot) / (denom + eps)

            tc = torch.clamp(tc_inf, 0.0, 1.0)
            sc = torch.clamp(sc_inf, 0.0, 1.0)
            
            dist_vec = w0 + tc.unsqueeze(-1) * u_i - sc.unsqueeze(-1) * v_j
            dist_sq = torch.sum(dist_vec * dist_vec, dim=-1)
            dists = torch.sqrt(dist_sq + eps)

            if self.knn_distance_threshold is not None:
                dists = dists.masked_fill(dists > self.knn_distance_threshold, float('inf'))

            if src_key_padding_mask is not None:
                 row_mask = src_key_padding_mask.unsqueeze(2).expand(-1, -1, seq_len)
                 dists = dists.masked_fill(row_mask, float('inf'))
                 col_mask = src_key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
                 dists = dists.masked_fill(col_mask, float('inf'))
                 diag_indices = torch.arange(seq_len, device=device)
                 dists[:, diag_indices, diag_indices] = dists[:, diag_indices, diag_indices].masked_fill(src_key_padding_mask, float('inf'))
                 dists[:, diag_indices, diag_indices] = dists[:, diag_indices, diag_indices].masked_fill(~src_key_padding_mask, 0.0)

            try:
                _, topk_indices = torch.topk(dists, k=k_actual, dim=-1, largest=False, sorted=False)
            except RuntimeError as e:
                print(f"Error during topk (Segment Distance): {e}")
                print(f"Dists shape: {dists.shape}, k_actual: {k_actual}")
                print(f"Dists has NaNs: {torch.isnan(dists).any()}")
                print(f"Dists has Infs: {torch.isinf(dists).any()}")
                if src_key_padding_mask is not None: print(f"Padding mask for row 0: {src_key_padding_mask[0,:] if bsz>0 else 'N/A'}")
                raise e

            attn_mask = torch.ones(bsz, seq_len, seq_len, dtype=torch.bool, device=device)
            attn_mask = attn_mask.scatter_(-1, topk_indices, False)

            if src_key_padding_mask is not None:
                attn_mask.masked_fill_(src_key_padding_mask.unsqueeze(2), True)
                attn_mask.masked_fill_(src_key_padding_mask.unsqueeze(1), True)

        embedded = self.input_embed(src) * math.sqrt(self.d_model)
        transformer_input = embedded

        transformer_output = self.transformer_encoder(
            src=transformer_input,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        logits = self.output_layer(transformer_output)
        return logits
