"""Decide whether the training step is GPU-bound or CPU-bound.

Runs the exact lowlight_train step (same loss/optimizer) on a fixed random batch,
timing (a) wall clock, (b) GPU busy time from CUDA events, (c) CPU time via time.process_time.
"""
import sys
import time

import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model  # noqa: E402
import Myloss  # noqa: E402

torch.backends.cudnn.benchmark = True
print("cuda ok:", torch.cuda.is_available(), flush=True)
print("gpu mem alloc:",
      torch.cuda.mem_get_info(), flush=True)

net = model.enhance_net_nopool().cuda().train()
L_color, L_spa, L_exp, L_TV = Myloss.L_color(), Myloss.L_spa(), Myloss.L_exp(16, 0.6), Myloss.L_TV()
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
x = torch.rand(8, 3, 256, 256).cuda()


def step():
    e1, enh, A = net(x)
    loss = (200 * L_TV(A) + torch.mean(L_spa(enh, x)) + 5 * torch.mean(L_color(enh))
            + 10 * torch.mean(L_exp(enh)))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


for _ in range(5):
    step()
torch.cuda.synchronize()

n = 20
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
wall0, cpu0 = time.perf_counter(), time.process_time()
ev0.record()
for _ in range(n):
    step()
ev1.record()
torch.cuda.synchronize()
wall = time.perf_counter() - wall0
cpu = time.process_time() - cpu0
gpu_busy = ev0.elapsed_time(ev1) / 1000.0
print(f"n={n}: wall={wall / n * 1000:.1f}ms/iter  cpu={cpu / n * 1000:.1f}ms/iter  "
      f"gpu_busy={gpu_busy / n * 1000:.1f}ms/iter", flush=True)

# same, but forward-only under no_grad
for _ in range(3):
    with torch.no_grad():
        net(x)
torch.cuda.synchronize()
wall0, cpu0 = time.perf_counter(), time.process_time()
ev0.record()
for _ in range(n):
    with torch.no_grad():
        net(x)
ev1.record()
torch.cuda.synchronize()
wall = time.perf_counter() - wall0
cpu = time.process_time() - cpu0
print(f"fwd-only: wall={wall / n * 1000:.1f}ms/iter  cpu={cpu / n * 1000:.1f}ms/iter  "
      f"gpu_busy={ev0.elapsed_time(ev1) / n:.1f}ms/iter", flush=True)

# single conv reference on the same data
conv = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
opt2 = torch.optim.Adam(conv.parameters(), lr=1e-4)
for _ in range(5):
    y = conv(x); y.sum().backward(); opt2.step(); opt2.zero_grad()
torch.cuda.synchronize()
wall0 = time.perf_counter()
ev0.record()
for _ in range(50):
    y = conv(x); y.sum().backward(); opt2.step(); opt2.zero_grad()
ev1.record()
torch.cuda.synchronize()
print(f"ref single conv fwd+bwd: {ev0.elapsed_time(ev1) / 50:.2f}ms gpu, "
      f"{(time.perf_counter() - wall0) / 50 * 1000:.2f}ms wall", flush=True)
