# refiner_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter
from typing import Optional, Tuple, List
import numpy as np
import math
import refiner_config as r_config

# Custom Submodules
class CustomMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim: raise ValueError(f"embed_dim must be divisible by num_heads")
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
            bsz, tgt_len, _ = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, _ = query.shape
            src_len, _, _ = key.shape
            query,key,value = query.transpose(0,1),key.transpose(0,1),value.transpose(0,1)

        q,k,v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale_factor = math.sqrt(self.head_dim)
        
        if scale_factor == 0: scale_factor = 1.0
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale_factor
        
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2 and key_padding_mask.shape[0] == bsz and key_padding_mask.shape[1] == src_len:
                key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(key_padding_mask_expanded.bool(), float('-inf'))
        
        if attn_mask is not None:
            attn_mask_expanded = None
            if attn_mask.dim() == 3 and attn_mask.shape[0] == bsz and attn_mask.shape[1] == tgt_len and attn_mask.shape[2] == src_len: attn_mask_expanded = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2 and attn_mask.shape[0] == tgt_len and attn_mask.shape[1] == src_len: attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 4 and attn_mask.shape == (bsz, self.num_heads, tgt_len, src_len): attn_mask_expanded = attn_mask
            if attn_mask_expanded is not None:
                try: attn_scores = attn_scores.masked_fill(attn_mask_expanded.bool(), float('-inf'))
                except RuntimeError as e_mask: print(f"  [MHA_ERROR] Error applying attn_mask. Error: {e_mask}")
        
        attn_probs = F.softmax(attn_scores + 1e-9, dim=-1)
        if torch.isnan(attn_probs).any(): attn_probs = attn_probs.nan_to_num(nan=0.0)
        attn_output = torch.matmul(attn_probs, v)
        if torch.isnan(attn_output).any(): attn_output = attn_output.nan_to_num(nan=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        if torch.isnan(attn_output).any(): attn_output = attn_output.nan_to_num(nan=0.0)
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
        self.dropout_attn_res = Dropout(dropout)
        self.dropout_ff_res = Dropout(dropout)
        
        if isinstance(activation, str): self.activation = F.relu if activation == "relu" else F.gelu
        else: self.activation = activation
        
    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        if torch.isnan(x).any(): return x.nan_to_num(nan=0.0)
        if self.norm_first:
            norm1_out = self.norm1(x)
            attn_out = self._sa_block(norm1_out, attn_mask, src_key_padding_mask)
            x = x + self.dropout_attn_res(attn_out)
            norm2_out = self.norm2(x)
            ff_out = self._ff_block(norm2_out)
            x = x + self.dropout_ff_res(ff_out)
        else:
            attn_out = self._sa_block(x, attn_mask, src_key_padding_mask)
            x = self.norm1(x + self.dropout_attn_res(attn_out))
            ff_out = self._ff_block(x)
            x = self.norm2(x + self.dropout_ff_res(ff_out))
        return x
    
    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if torch.isnan(attn_output).any(): attn_output = attn_output.nan_to_num(nan=0.0)
        return attn_output
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CustomTransformerEncoder(Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer_template, num_layers, norm=None):
        super().__init__()
        self.layers = ModuleList([
            CustomTransformerEncoderLayer(
                d_model=encoder_layer_template.self_attn.embed_dim, nhead=encoder_layer_template.self_attn.num_heads,
                dim_feedforward=encoder_layer_template.linear1.out_features, dropout=encoder_layer_template.dropout.p,
                activation=encoder_layer_template.activation, layer_norm_eps=encoder_layer_template.norm1.eps,
                batch_first=encoder_layer_template.batch_first, norm_first=encoder_layer_template.norm_first
            ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, attn_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if torch.isnan(output).any(): output = output.nan_to_num(nan=0.0)
        if self.norm is not None:
            output = self.norm(output)
            if torch.isnan(output).any(): output = output.nan_to_num(nan=0.0)
        return output


# Helper function to create MLPs
def _create_mlp(input_dim: int, output_dim: int, hidden_dims: Optional[List[int]], dropout: float) -> nn.Module:
    """Helper to create a linear layer or an MLP."""
    if not hidden_dims:
        return nn.Linear(input_dim, output_dim)
    
    layers = []
    current_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        current_dim = h_dim
    
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


# RefinerTransformerEdgeClassifier
class RefinerTransformerEdgeClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward,
                 num_classes, dropout=0.1, norm_first=False,
                 k_nearest: Optional[int] = None,
                 knn_distance_threshold: Optional[float] = None,
                 # MLP Head dimension arguments
                 refiner_main_head_dims: Optional[List[int]] = None,
                 refiner_correctness_head_dims: Optional[List[int]] = None):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.k_nearest = k_nearest
        self.knn_distance_threshold = knn_distance_threshold
        self.input_dim = input_dim

        if d_model % nhead != 0: raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_embed = nn.Linear(self.input_dim, d_model)

        custom_encoder_layer_template = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=norm_first
        )
        encoder_norm = LayerNorm(d_model, eps=custom_encoder_layer_template.norm1.eps) if not norm_first else None
        self.transformer_encoder = CustomTransformerEncoder(custom_encoder_layer_template, num_encoder_layers, norm=encoder_norm)

        self.main_classification_head = _create_mlp(
            input_dim=d_model,
            output_dim=num_classes,
            hidden_dims=refiner_main_head_dims,
            dropout=dropout
        )

        self.correctness_head = _create_mlp(
            input_dim=d_model,
            output_dim=1,
            hidden_dims=refiner_correctness_head_dims,
            dropout=dropout
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        if self.input_embed.bias is not None: self.input_embed.bias.data.zero_()

        # Initialize weights for sequential MLP heads
        for head in [self.main_classification_head, self.correctness_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.uniform_(-initrange, initrange)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def _generate_knn_attention_mask(self, src_features: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]):
        bsz, seq_len, total_feature_dim = src_features.shape
        device = src_features.device

        if self.k_nearest is None or self.k_nearest <= 0 or seq_len <= 1:
            return None

        k_actual = min(self.k_nearest, seq_len - 1)
        if k_actual <= 0: return None

        try:
            x1_idx = r_config.GEOM_FEATURE_COLS.index('x1_norm')
            y1_idx = r_config.GEOM_FEATURE_COLS.index('y1_norm')
            x2_idx = r_config.GEOM_FEATURE_COLS.index('x2_norm')
            y2_idx = r_config.GEOM_FEATURE_COLS.index('y2_norm')
        except ValueError:
             print(f"  [MODEL_KNN_ERROR] Critical geometric coordinate feature not found in r_config.GEOM_FEATURE_COLS. KNN mask cannot be built.")
             return None

        num_geom_features = len(r_config.GEOM_FEATURE_COLS)
        if total_feature_dim < num_geom_features:
            print(f"  [MODEL_KNN_ERROR] src_features dim ({total_feature_dim}) < num_geom_features ({num_geom_features}). KNN mask cannot be built.")
            return None

        geom_block = src_features[:, :, :num_geom_features]
        p1 = geom_block[:, :, [x1_idx, y1_idx]]
        q1 = geom_block[:, :, [x2_idx, y2_idx]]
        
        if torch.isnan(p1).any() or torch.isinf(p1).any() or torch.isnan(q1).any() or torch.isinf(q1).any(): return None

        u = q1 - p1

        if torch.isnan(u).any() or torch.isinf(u).any(): return None
        
        p1_i = p1.unsqueeze(2)
        u_i = u.unsqueeze(2)
        p2_j = p1.unsqueeze(1)
        v_j = u.unsqueeze(1)
        w0 = p1_i - p2_j
        a = torch.sum(u_i * u_i, dim=-1)
        b_val = torch.sum(u_i * v_j, dim=-1)
        c = torch.sum(v_j * v_j, dim=-1)
        d_param = torch.sum(u_i * w0, dim=-1)
        e_param = torch.sum(v_j * w0, dim=-1)
        if torch.isnan(a).any() or torch.isnan(b_val).any() or torch.isnan(c).any() or torch.isnan(d_param).any() or torch.isnan(e_param).any(): return None
        
        eps_dist = 1e-7
        denom = a * c - b_val * b_val
        if torch.isnan(denom).any(): return None
        sign_denom = torch.sign(denom)
        sign_denom_no_zero = torch.where(denom == 0, torch.tensor(1.0, device=denom.device, dtype=denom.dtype), sign_denom)
        fill_values = eps_dist * sign_denom_no_zero
        safe_denom_intermediate = denom + eps_dist * sign_denom_no_zero
        condition_mask = torch.abs(safe_denom_intermediate) < (eps_dist / 2.0)
        safe_denom = torch.where(condition_mask, fill_values, safe_denom_intermediate)
        
        tc = (b_val * e_param - c * d_param) / safe_denom
        sc = (a * e_param - b_val * d_param) / safe_denom
        if torch.isnan(sc).any() or torch.isnan(tc).any(): return None
        
        sc = torch.clamp(sc, 0.0, 1.0)
        tc = torch.clamp(tc, 0.0, 1.0)
        dist_vec = w0 + u_i * tc.unsqueeze(-1) - v_j * sc.unsqueeze(-1)
        
        if torch.isnan(dist_vec).any(): return None
        
        dists = torch.norm(dist_vec, p=2, dim=-1)
        
        if torch.isnan(dists).any(): return None

        if self.knn_distance_threshold is not None: dists = dists.masked_fill(dists > self.knn_distance_threshold, float('inf'))
        
        diag_indices = torch.arange(seq_len, device=device)
        dists[:, diag_indices, diag_indices] = float('inf')
        
        if src_key_padding_mask is not None:
             dists.masked_fill_(src_key_padding_mask.unsqueeze(2).bool(), float('inf'))
             dists.masked_fill_(src_key_padding_mask.unsqueeze(1).bool(), float('inf'))

        k_for_topk = k_actual
        
        try:
            if k_for_topk < 1: return None
            _, topk_indices = torch.topk(dists, k=k_for_topk, dim=-1, largest=False, sorted=False)
        except RuntimeError:
            num_valid_neighbors = (~torch.isinf(dists)).sum(dim=-1)
            min_valid = num_valid_neighbors.min().item()
            if k_for_topk > min_valid and min_valid > 0:
                k_for_topk = min_valid
                if k_for_topk < 1: return None
                _, topk_indices = torch.topk(dists, k=k_for_topk, dim=-1, largest=False, sorted=False)
            else: return None

        attn_mask_knn = torch.ones(bsz, seq_len, seq_len, dtype=torch.bool, device=device)
        scatter_src_false = torch.zeros_like(topk_indices, dtype=torch.bool)
        attn_mask_knn.scatter_(dim=2, index=topk_indices, src=scatter_src_false)
        return attn_mask_knn


    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        if torch.isnan(src).any() or torch.isinf(src).any():
            bs, sl = src.shape[0], src.shape[1]
            nan_main_logits = torch.full((bs, sl, self.num_classes), float('nan'), device=src.device, dtype=src.dtype)
            nan_corr_logits = torch.full((bs, sl, 1), float('nan'), device=src.device, dtype=src.dtype)
            return nan_main_logits, nan_corr_logits

        attn_mask_knn = self._generate_knn_attention_mask(src, src_key_padding_mask)
        embedded = self.input_embed(src)
        if torch.isnan(embedded).any() or torch.isinf(embedded).any():
            bs, sl = embedded.shape[0], embedded.shape[1]
            nan_main_logits = torch.full((bs, sl, self.num_classes), float('nan'), device=embedded.device, dtype=embedded.dtype)
            nan_corr_logits = torch.full((bs, sl, 1), float('nan'), device=embedded.device, dtype=embedded.dtype)
            return nan_main_logits, nan_corr_logits

        transformer_output = self.transformer_encoder(
            src=embedded, mask=attn_mask_knn, src_key_padding_mask=src_key_padding_mask
        )
        if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
            bs, sl = transformer_output.shape[0], transformer_output.shape[1]
            nan_main_logits = torch.full((bs, sl, self.num_classes), float('nan'), device=transformer_output.device, dtype=transformer_output.dtype)
            nan_corr_logits = torch.full((bs, sl, 1), float('nan'), device=transformer_output.device, dtype=transformer_output.dtype)
            return nan_main_logits, nan_corr_logits

        main_logits = self.main_classification_head(transformer_output)
        correctness_logits = self.correctness_head(transformer_output)

        return main_logits, correctness_logits
