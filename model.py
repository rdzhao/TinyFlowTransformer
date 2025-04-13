import torch
import torch as torch.nn
import torch.nn.functional as F

import einops

import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_embd, n_head, causal=False):
        super().__init__()
        assert d_embd % n_head == 0
        
        self.d_emdb = d_embd
        self.n_head = n_head
        self.causal = causal

        self.d_head = self.d_embd // self.n_head

        self.qkv = nn.Linear(self.d_embd, 3*self.d_embd)
        self.proj = nn.Linear(self.d_embd, self.d_embd)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = einops.rearrange(q, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        k = einops.rearrange(k, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        v = einops.rearrange(v, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        
        if self.causal:
            w = q @ k.transpose(-2, -1)
            mask = torch.tril(torch.ones((t, t)), device=x.device)
            w = torch.mask_fill(w, mask == 0, torch.float("-inf"))
            w /= math.sqrt(self.d_head)
            w = F.softmax(w, dim=-1)
            o = w @ v
        else:
            w = q @ k.transpose(-2, -1)
            w /= math.sqrt(self.d_head)
            w = F.softmax(w, dim=-1)
            o = w @ v
        
        o = einops.rearrange(q, "b n_head t d_head -> b t (n_head d_head)")
        o = self.proj(o)

        return o


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_embd, n_head, n_cond_embd):
        super().__init__()
        assert d_embd % n_head == 0

        self.d_embd = d_embd
        self.n_head = n_head
        self.n_cond_embd = n_cond_embd
        
        self.d_head = self.d_embd // self.n_head

        self.q = nn.Linear(self.n_embd, self.n_embd)
        self.kv = nn.Linear(self.n_cond_embd, 2*self.n_embd)

        self.proj = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x, cond):
        b, t, c = x.shape
        b_cond, t_cond, c_cond = cond.shape

        q = self.q(x)
        kv = self.kv(cond)

        k, v = torch.chunk(kv, chunks=2, dim=-1)

        q = einops.rearrange(q, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        k = einops.rearrange(k, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)
        v = einops.rearrange(v, "b t (n_head d_head) -> b n_head t d_head", n_head=self.n_head, d_head=self.d_head)

        w = q @ k.transpose(-2, -1)
        w /= math.sqrt(self.d_head)
        w = F.softmax(w, dim=-1)
        o = w @ v

        o = einops.rearrange(o, "b n_head t d_head -> b t (n_head d_head)")
        o = self.proj(o)

        return o