import torch

device = torch.device("cuda")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

c = a @ b

torch.cuda.synchronize()

print("Device:", c.device)
print("Result shape:", c.shape)
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")