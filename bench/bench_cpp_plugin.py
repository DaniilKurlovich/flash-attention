import sys, torch
from src.flash_attention import attention_tiled_online_softmax_cpp

torch.cuda.synchronize()
q = torch.randn(4, 8, 512, 64, dtype=torch.float16, device="cuda")
k = torch.randn(4, 8, 512, 64, dtype=torch.float16, device="cuda")
v = torch.randn(4, 8, 512, 64, dtype=torch.float16, device="cuda")

for _ in range(5):
    out = attention_tiled_online_softmax_cpp(q, k, v)
torch.cuda.synchronize()

out = attention_tiled_online_softmax_cpp(q, k, v)
torch.cuda.synchronize()
print("done", out.shape)
