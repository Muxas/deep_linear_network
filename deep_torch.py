import torch
from deep_linear_network import DeepLinear

x = torch.randn(1, 1024, 256)
linear = DeepLinear(256, 256, 256, 256, 128) # w1-w3 (256 x 256) w4 (256 x 128)
linear(x) # (1, 1024, 128)
