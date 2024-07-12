import torch

mix_prob = torch.tensor([0.5])
mix_prob.share_memory_()

def change_mix_prob(p):
    mix_prob[0] = p
def get_mix_prob():
    return mix_prob[0].item()