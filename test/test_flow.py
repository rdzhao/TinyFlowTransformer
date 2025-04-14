from model import FlowModel

import torch

def test_flow():
    n_blocks = 3
    b = 2
    h_latent = 16
    w_latent = 16
    c_latent = 4

    d_embd = 8
    n_head = 2

    c_time = 12
    
    t_cond = 7
    c_cond = 13

    patch_size = 2

    x = torch.rand((b, c_latent, h_latent, w_latent))
    time = torch.rand((b))
    cond = torch.rand((b, t_cond, c_cond))

    flow = FlowModel(n_blocks, c_latent, d_embd, n_head, c_time, c_cond, patch_size)
    o = flow(x, time, cond)


if __name__ == "__main__":
    test_flow()