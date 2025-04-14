from model import ModulatedTransformerBlock

import torch

def test_block():
    b = 2
    t = 3
    c = 6
    n_head = 2

    x = torch.rand((b, t, c))

    c_time = 19
    
    time = torch.rand(b, c_time)

    t_cond = 13
    c_cond = 27

    cond = torch.rand((b, t_cond, c_cond))

    block = ModulatedTransformerBlock(c, n_head, c_time, c_cond)
    x = block(x, time, cond)

    print(x)

if __name__ == "__main__":
    test_block()