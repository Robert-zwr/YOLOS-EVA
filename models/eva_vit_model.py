# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

#from .transformer import PatchDropout
from .layers.rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast

import numpy as np

if os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint
'''
try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")
'''

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,

        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    
class SwiGLU_sep(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False, num_det_tokens=100):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1_patch = nn.Linear(in_features, hidden_features)
        self.w1_det = nn.Linear(in_features, hidden_features)
        self.w2_patch = nn.Linear(in_features, hidden_features)
        self.w2_det = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln_patch = norm_layer(hidden_features) if subln else nn.Identity()
        self.ffn_ln_det = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3_patch = nn.Linear(hidden_features, out_features)
        self.w3_det = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

        self.num_det_tokens = num_det_tokens

    def forward(self, x):
        x_patch = x[:,:-self.num_det_tokens,:]
        x_det = x[:,-self.num_det_tokens:,:]
        # x1 = self.w1(x)
        x1_patch = self.w1_patch(x_patch)
        x1_det = self.w1_det(x_det)
        # x2 = self.w2(x)
        x2_patch = self.w2_patch(x_patch)
        x2_det = self.w2_det(x_det)
        # hidden = self.act(x1) * x2
        hidden_patch = self.act(x1_patch) * x2_patch
        hidden_det = self.act(x1_det) * x2_det
        # x = self.ffn_ln(hidden)
        x_patch = self.ffn_ln_patch(hidden_patch)
        x_det = self.ffn_ln_det(hidden_det)
        # x = self.w3(x)
        x_patch = self.w3_patch(x_patch)
        x_det = self.w3_det(x_det)
        x = torch.cat((x_patch, x_det), dim=1)
        x = self.drop(x)
        return x


def memory_efficient_attention_pytorch(query, key, value, attn_bias=None, p=0., scale=None, attn_mask=None):
    # query     [batch, seq_len, n_head, head_dim]
    # key       [batch, seq_len, n_head, head_dim]
    # value     [batch, seq_len, n_head, head_dim]
    # attn_bias [batch, n_head, seq_len, seq_len]

    if scale is None:
        scale = 1 / query.shape[-1] ** 0.5
    
    # BLHC -> BHLC
    #query = query.transpose(1, 2)
    #key = key.transpose(1, 2)
    #value = value.transpose(1, 2)

    query = query * scale
    # BHLC @ BHCL -> BHLL
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias

    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        attn = attn.masked_fill(~attn_mask, float("-inf"))

    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    # BHLL @ BHLC -> BHLC
    out = attn @ value
    # BHLC -> BLHC
    out = out.transpose(1, 2)
    return out

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,proj_drop=0., window_size=None, 
            attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm, num_det_tokens=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_det_tokens = num_det_tokens
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            # np.savetxt('q.csv',q[0].cpu().detach().numpy(),fmt='%.4f',delimiter=',')
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else: 

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:-self.num_det_tokens, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t, q[:, :, -self.num_det_tokens:, :]), -2).type_as(v)

            k_t = k[:, :, 1:-self.num_det_tokens, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t, k[:, :, -self.num_det_tokens:, :]), -2).type_as(v)

        if self.xattn:
            #q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            #k = k.permute(0, 2, 1, 3)
            #v = v.permute(0, 2, 1, 3)

            #x = xops.memory_efficient_attention(
            #    q, k, v,
            #    p=self.xattn_drop,
            #    scale=self.scale,
            #    )
            x = memory_efficient_attention_pytorch(q, k, v, p=self.xattn_drop, scale=self.scale, attn_mask=attn_mask)

            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
    
class Attention_sep(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,proj_drop=0., window_size=None, 
            attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm, num_det_tokens=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_det_tokens = num_det_tokens
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj_patch = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj_patch = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj_patch = nn.Linear(dim, all_head_dim, bias=False)
            self.q_proj_det = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj_det = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj_det = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv_patch = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.qkv_det = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias_patch = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias_patch = nn.Parameter(torch.zeros(all_head_dim))
            self.q_bias_det = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias_det = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias_patch = None
            self.v_bias_patch = None
            self.q_bias_det = None
            self.v_bias_det = None

        if window_size:
            raise NotImplementedError
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj_patch = nn.Linear(all_head_dim, dim)
        self.proj_det = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q_patch = F.linear(input=x[:,:-self.num_det_tokens,:], weight=self.q_proj_patch.weight, bias=self.q_bias_patch)
            k_patch = F.linear(input=x[:,:-self.num_det_tokens,:], weight=self.k_proj_patch.weight, bias=None)
            v_patch = F.linear(input=x[:,:-self.num_det_tokens,:], weight=self.v_proj_patch.weight, bias=self.v_bias_patch)

            q_det = F.linear(input=x[:,-self.num_det_tokens:,:], weight=self.q_proj_det.weight, bias=self.q_bias_det)
            k_det = F.linear(input=x[:,-self.num_det_tokens:,:], weight=self.k_proj_det.weight, bias=None)
            v_det = F.linear(input=x[:,-self.num_det_tokens:,:], weight=self.v_proj_det.weight, bias=self.v_bias_det)

            q = torch.cat((q_patch, q_det), dim=1)
            k = torch.cat((k_patch, k_det), dim=1)
            v = torch.cat((v_patch, v_det), dim=1)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else: 
            qkv_bias_patch = None
            qkv_bias_det = None
            if self.q_bias_patch is not None:
                qkv_bias_patch = torch.cat((self.q_bias_patch, torch.zeros_like(self.v_bias_patch, requires_grad=False), self.v_bias_patch))
            if self.q_bias_det is not None:
                qkv_bias_det = torch.cat((self.q_bias_det, torch.zeros_like(self.v_bias_det, requires_grad=False), self.v_bias_det))
            
            qkv_patch = F.linear(input=x[:,:-self.num_det_tokens,:], weight=self.qkv_patch.weight, bias=qkv_bias_patch)
            qkv_det = F.linear(input=x[:,-self.num_det_tokens:,:], weight=self.qkv_det.weight, bias=qkv_bias_det)
            qkv = torch.cat((qkv_patch, qkv_det), dim=1)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:-self.num_det_tokens, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t, q[:, :, -self.num_det_tokens:, :]), -2).type_as(v)

            k_t = k[:, :, 1:-self.num_det_tokens, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t, k[:, :, -self.num_det_tokens:, :]), -2).type_as(v)

        if self.xattn:
            #q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            #k = k.permute(0, 2, 1, 3)
            #v = v.permute(0, 2, 1, 3)

            #x = xops.memory_efficient_attention(
            #    q, k, v,
            #    p=self.xattn_drop,
            #    scale=self.scale,
            #    )
            x = memory_efficient_attention_pytorch(q, k, v, p=self.xattn_drop, scale=self.scale, attn_mask=attn_mask)

            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x_patch = self.proj_patch(x[:,:-self.num_det_tokens,:])
            x_det = self.proj_det(x[:,-self.num_det_tokens:,:])
            x = torch.cat((x_patch, x_det), dim=1)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False, num_det_tokens=100, partial_finetune=False, partial_finetune_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if partial_finetune_attn:
            self.attn = Attention_sep(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer, num_det_tokens=num_det_tokens)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer, num_det_tokens=num_det_tokens)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            if partial_finetune:
                self.mlp = SwiGLU_sep(
                    in_features=dim, 
                    hidden_features=mlp_hidden_dim, 
                    subln=subln,
                    norm_layer=norm_layer,
                    num_det_tokens=num_det_tokens,
                )
            else:
                self.mlp = SwiGLU(
                    in_features=dim, 
                    hidden_features=mlp_hidden_dim, 
                    subln=subln,
                    norm_layer=norm_layer,
                )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # [1, 14*14, vision_cfg.width:768]
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_det_tokens=100, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,use_abs_pos_emb=True, use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, rope=False,use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, 
                 xattn=False, postnorm=False,pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False, attn_mask=False, 
                 partial_finetune=False, partial_finetune_attn=False):
        super().__init__()
        #self.img_size = img_size
        if isinstance(img_size,tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.num_det_tokens = num_det_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu
        self.partial_finetune = partial_finetune
        self.partial_finetune_attn = partial_finetune_attn

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu, 
                num_det_tokens = self.num_det_tokens, partial_finetune=partial_finetune, partial_finetune_attn = partial_finetune_attn)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing
        self.has_mid_pe = False

        self.attn_mask = attn_mask

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if self.partial_finetune_attn:
                rescale(layer.attn.proj_patch.weight.data, layer_id + 1)
                rescale(layer.attn.proj_det.weight.data, layer_id + 1)
            else:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                if self.partial_finetune:
                    rescale(layer.mlp.w3_patch.weight.data, layer_id + 1)
                    rescale(layer.mlp.w3_det.weight.data, layer_id + 1)
                else:
                    rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def lock(self, unlocked_groups=0):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def finetune_det(self, img_size=[800, 1344], mid_pe_size=None, use_checkpoint=False):
        # import pdb;pdb.set_trace()

        import math
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        # self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, self.num_det_tokens, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        det_pos_embed = torch.zeros(1, self.num_det_tokens, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        # torch.Size([1, 768, 50, 84])
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1,2) # torch.Size([1, 4200, 768])
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos_embed, patch_pos_embed), dim=1), requires_grad=not (self.partial_finetune or self.partial_finetune_attn))
        self.det_pos_embed = torch.nn.Parameter(det_pos_embed)

        _, head_dim = self.rope.freqs_cos.shape
        freqs_cos = self.rope.freqs_cos.transpose(0,1).view(B, head_dim, P_H, P_W)
        freqs_sin = self.rope.freqs_sin.transpose(0,1).view(B, head_dim, P_H, P_W)
        freqs_cos = nn.functional.interpolate(freqs_cos, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False) #torch.Size([1, 64, 50, 84])
        freqs_sin = nn.functional.interpolate(freqs_sin, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        self.register_buffer("freqs_cos", freqs_cos.squeeze().flatten(1).transpose(0,1))
        self.register_buffer("freqs_sin", freqs_sin.squeeze().flatten(1).transpose(0,1))

        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print('No mid pe')
        else:
            print('Has mid pe')
            self.mid_pos_embed = nn.Parameter(torch.zeros(self.depth - 1, 1, 1 + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2) + self.num_det_tokens, self.embed_dim))
            trunc_normal_(self.mid_pos_embed, std=.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        self.use_checkpoint=use_checkpoint

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        _, head_dim = self.rope.freqs_cos.shape

        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        patch_pos_embed = pos_embed[:, 1:, :] # [1,4200,768]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        freqs_cos = self.freqs_cos.transpose(0,1)
        freqs_sin = self.freqs_sin.transpose(0,1)
        B, E, Q = patch_pos_embed.shape


        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        freqs_cos = freqs_cos.view(B, head_dim, P_H, P_W)
        freqs_sin = freqs_sin.view(B, head_dim, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        freqs_cos = nn.functional.interpolate(freqs_cos, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        freqs_sin = nn.functional.interpolate(freqs_sin, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        self.rope.freqs_cos = freqs_cos.squeeze().flatten(1).transpose(0,1)
        self.rope.freqs_sin = freqs_sin.squeeze().flatten(1).transpose(0,1)
        
        return scale_pos_embed
    
    def get_rect_mask(self, patch_seq_len=196):

        total_seq_len = 1 + patch_seq_len + self.num_det_tokens
        mask = torch.ones(size=[total_seq_len, total_seq_len], device='cuda')
        mask[:-self.num_det_tokens, -self.num_det_tokens:] = 0

        return mask
    
    def forward_features(self, x, return_all_features=False):
        
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        
        x = self.patch_embed(x)  # [B,3,H,W] -> torch.Size([B, n_patch, embed_dim=768])
        batch_size, seq_len, _ = x.size()

        # interpolate init pe
        if (self.pos_embed.shape[1] - 1) != seq_len:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
            self.rope.freqs_cos = self.freqs_cos
            self.rope.freqs_sin = self.freqs_sin

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)  # [B,1+n_patch+100, embed_dim=768]
        if self.pos_embed is not None:
            x = x + torch.cat((temp_pos_embed, self.det_pos_embed), dim=1)
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)  # Identity()
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None  # None
        attn_mask = self.get_rect_mask(patch_seq_len=seq_len) if self.attn_mask else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias, attn_mask))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)

        if not return_all_features:  # x:[B, 1+n_patch+100, embed_dim=768]
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, -self.num_det_tokens:, :]  # torch.Size([1, 100, 768])
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        # x = self.head(x) # [1,512]
        return x