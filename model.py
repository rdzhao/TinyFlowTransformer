import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_embd, n_head, causal=False, flash=False):
        super().__init__()
        assert d_embd % n_head == 0
        
        self.d_embd = d_embd
        self.n_head = n_head
        self.causal = causal
        self.flash = flash

        self.d_head = self.d_embd // self.n_head

        self.qkv = nn.Linear(self.d_embd, 3*self.d_embd)
        self.proj = nn.Linear(self.d_embd, self.d_embd)

    def use_flash(self, use_flash):
            self.flash = use_flash

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = einops.rearrange(q, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        k = einops.rearrange(k, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        v = einops.rearrange(v, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        
        if not self.flash:
            if self.causal:
                w = q @ k.transpose(-2, -1)
                mask = torch.tril(torch.ones((t, t), device=x.device))
                w = torch.masked_fill(w, mask == 0, float("-inf"))
                w /= math.sqrt(self.d_head)
                w = F.softmax(w, dim=-1)
                o = w @ v
            else:
                w = q @ k.transpose(-2, -1)
                w /= math.sqrt(self.d_head)
                w = F.softmax(w, dim=-1)
                o = w @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        
        o = einops.rearrange(q, "b n_head t d_head -> b t (n_head d_head)")
        o = self.proj(o)

        return o


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_embd, n_head, d_cond_embd, flash=False):
        super().__init__()
        assert d_embd % n_head == 0

        self.d_embd = d_embd
        self.n_head = n_head
        self.d_cond_embd = d_cond_embd
        self.flash = flash
        
        self.d_head = self.d_embd // self.n_head

        self.q = nn.Linear(self.d_embd, self.d_embd)
        self.kv = nn.Linear(self.d_cond_embd, 2*self.d_embd)

        self.proj = nn.Linear(self.d_embd, self.d_embd)

    def use_flash(self, use_flash):
        self.flash = use_flash

    def forward(self, x, cond):
        b, t, c = x.shape
        b_cond, t_cond, c_cond = cond.shape

        q = self.q(x)
        kv = self.kv(cond)

        k, v = torch.chunk(kv, chunks=2, dim=-1)

        q = einops.rearrange(q, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        k = einops.rearrange(k, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        v = einops.rearrange(v, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)

        if not self.flash:
            w = q @ k.transpose(-2, -1)
            w /= math.sqrt(self.d_head)
            w = F.softmax(w, dim=-1)
            o = w @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v)

        o = einops.rearrange(o, "b n_head t d_head -> b t (n_head d_head)")
        o = self.proj(o)

        return o

class ModulatedTransformerBlock(nn.Module):
    def __init__(self, d_embd, n_head,d_time_embd, d_cond_embd):
        super().__init__()
        assert d_embd % n_head == 0

        self.d_embd = d_embd
        self.n_head = n_head
        self.d_time_embd = d_time_embd
        self.d_cond_embd = d_cond_embd

        self.d_head = self.d_embd // self.n_head
        
        self.layernorm_1 = nn.LayerNorm(self.d_embd)
        self.self_attention = MultiHeadSelfAttention(self.d_embd, self.n_head)
        self.scale_alpha_1 = nn.Linear(self.d_time_embd, self.d_embd)
        self.shift_beta_1 = nn.Linear(self.d_time_embd, self.d_embd)
        self.scale_gamma_1 = nn.Linear(self.d_time_embd, self.d_embd)

        self.layernorm_2 = nn.LayerNorm(self.d_embd)
        self.cross_attention = MultiHeadCrossAttention(self.d_embd, self.n_head, self.d_cond_embd)
        self.scale_alpha_2 = nn.Linear(self.d_time_embd, self.d_embd)
        self.shift_beta_2 = nn.Linear(self.d_time_embd, self.d_embd)
        self.scale_gamma_2 = nn.Linear(self.d_time_embd, self.d_embd)

        self.layernorm_3 = nn.LayerNorm(self.d_embd)
        self.ffn = nn.Linear(self.d_embd, self.d_embd)
        self.scale_alpha_3 = nn.Linear(self.d_time_embd, self.d_embd)
        self.shift_beta_3 = nn.Linear(self.d_time_embd, self.d_embd)
        self.scale_gamma_3 = nn.Linear(self.d_time_embd, self.d_embd)

    def forward(self, x, time, cond):
        b, t, c = x.shape # [B, T, C]
        b_time, c_time = time.shape # [B, C]
        b_cond, t_cond, c_cond = cond.shape # [B, T ,C]

        time = einops.rearrange(time, "b c -> b 1 c")
        sg_1 = self.scale_gamma_1(time)
        sb_1 = self.shift_beta_1(time)
        sa_1 = self.scale_alpha_1(time)
        x = x + self.self_attention(self.layernorm_1(x) * sg_1 + sb_1) * sa_1

        sg_2 = self.scale_gamma_2(time)
        sb_2 = self.shift_beta_2(time)
        sa_2 = self.scale_alpha_2(time)
        x = x + self.cross_attention(self.layernorm_2(x) * sg_2 + sb_2, cond) * sa_2

        sg_3 = self.scale_gamma_3(time)
        sb_3 = self.shift_beta_3(time)
        sa_3 = self.scale_alpha_3(time)
        x = x + self.ffn(self.layernorm_3(x) * sg_3 + sb_3) * sa_3

        return x

class FlowModel(nn.Module):
    def __init__(self, n_blocks, c_latent, d_embd, n_head, d_cond_embd, d_time_embd, patch_size):
        super().__init__()
        self.n_blocks = n_blocks
        self.c_latent = c_latent
        self.d_embd = d_embd
        self.d_cond_embd = d_cond_embd
        self.d_time_embd = d_time_embd
        self.patch_hw = patch_hw

        self.patchify_conv = nn.Conv2d(self.c_latent, self.d_embd, kernel_size=self.patch_size, stride=self.patch_size)
        self.unpatchify_conv = nn.ConvTranspose2d(self.d_embd, self.c_latent, kernel_size=self.patch_size, stride=self.patch_size)

        self.time_module = nn.ModuleList([
            nn.Linear(self.d_time_embd, 3*self.d_time_embd),
            nn.GeLU(),
            nn.Linear(3*self.d_time_embd, self.d_time_embd),
        ])

        self.blocks = nn.ModuleList([
            ModulatedTransformerBlock(
                d_embd=self.d_embd,
                n_head=self.n_head,
                d_cond_embd=self.d_cond_embd,
                d_time_embd=self.d_time_embd
            )
            for _ in range(self.n_blocks)
        ])
        

    def patchify(self, x):
        x = self.patchify_conv(x)
        return x
    
    def unpatchify(self, x):
        x = self.unpatchify_conv(x)
        return x

    def sinusoidal_positional_embedding(self, seq_len, d_embd, device):
        embd = torch.zeros((seq_len, d_embd), device=device)
        pos = torch.arange(seq_len, device=device)
        embd[:, 0::2] = torch.sin(pos/torch.pow(10000, 2*pos / d_embd)).to(device)
        embd[:, 1::2] = torch.cos(pos/torch.pow(10000, 2*pos / d_embd)).to(device)
        return embd

    def time_embedding(self, time, d_time_embd, device):
        b = time.shape
        embd = torch.zeros((b, d_time_embd))
        pos = torch.arange(b, device=device)
        embd[:, 0::2] = torch.sin(pos/torch.pow(10000, 2*pos / d_time_embd)).to(device)
        embd[:, 1::2] = torch.cos(pos/torch.pow(10000, 2*pos / d_time_embd)).to(device)
        return embd

    def forward(self, x, time, cond):
        b, c, h, w = x.shape
        
        x = self.patchify(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        
        pos_embd = self.sinusoidal_positional_embedding(x.size(-2), self.d_embd, x.device)
        pos_embd = einops.rearrange(pos_embd, "t c -> 1 t c")
        
        x = x + pos_embd # [B, T, d_embd]

        time_embd = self.time_embedding(time, self.d_time_embd, x.device) # [B, d_time_embd]
        time_embd = self.time_module(time_embd)
        
        for block in self.blocks:
            x = block(x, time_embd, cond)

        x = F.layernorm(x)

        x = self.unpatchify(x)

        return x
        