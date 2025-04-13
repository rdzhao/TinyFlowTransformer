from model import MultiHeadSelfAttention, MultiHeadCrossAttention

import torch

import einops

def test_attention():
    b = 2
    t = 3
    c = 4
    n_head = 2
    x = torch.rand((b, t, c))

    self_attention = MultiHeadSelfAttention(c, n_head)
    self_attention.use_flash(True)
    x_flash_self= self_attention(x)
    self_attention.use_flash(False)
    x_scratch_self = self_attention(x)

    assert torch.allclose(x_flash_self, x_scratch_self, rtol=1e-05, atol=1e-08)

    self_attention_causal = MultiHeadSelfAttention(c, n_head, causal=True)
    self_attention_causal.use_flash(True)
    x_flash_self_causal= self_attention_causal(x)
    self_attention_causal.use_flash(False)
    x_scratch_self_causal = self_attention_causal(x)

    assert torch.allclose(x_flash_self_causal, x_scratch_self_causal, rtol=1e-05, atol=1e-08)

    t_cond = 5
    c_cond = 9
    cond = torch.rand((b, t_cond, c_cond))
    
    cross_attention = MultiHeadCrossAttention(c, n_head, c_cond)
    cross_attention.use_flash(True)
    x_flash_cross = cross_attention(x, cond)
    cross_attention.use_flash(False)
    x_scratch_cross = cross_attention(x, cond)
    
    assert torch.allclose(x_flash_cross, x_scratch_cross, rtol=1e-05, atol=1e-08)

if __name__ == "__main__":
    test_attention()