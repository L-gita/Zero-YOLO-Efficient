"""Profile where training time goes: GPU forward/backward vs dataloader."""
import sys, time
import torch
sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model, Myloss, dataloader

torch.backends.cudnn.benchmark = True
net = model.enhance_net_nopool().cuda().train()
L_color, L_spa = Myloss.L_color(), Myloss.L_spa()
L_exp, L_TV = Myloss.L_exp(16, 0.6), Myloss.L_TV()
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
x = torch.rand(8, 3, 256, 256).cuda()

for warm in range(3):
    e1, e2, A = net(x)
    loss = 200 * L_TV(A) + torch.mean(L_spa(e2, x)) + 5 * torch.mean(L_color(e2)) + 10 * torch.mean(L_exp(e2))
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()

import copy
n = 50
t0 = time.time()
for _ in range(n):
    e1, e2, A = net(x)
    loss = 200 * L_TV(A) + torch.mean(L_spa(e2, x)) + 5 * torch.mean(L_color(e2)) + 10 * torch.mean(L_exp(e2))
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
gpu = (time.time() - t0) / n * 1000

ds = dataloader.lowlight_loader(r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code\data\train_data_sice/")
torch.manual_seed(0)
t0 = time.time()
for i in range(n):
    ds[i]
cpu1 = (time.time() - t0) / n * 1000
print(f"gpu fwd+bwd per iter : {gpu:.1f} ms")
print(f"dataset __getitem__  : {cpu1:.1f} ms   (1 worker)")
print(f"ratio load/compute   : {cpu1/gpu:.2f}x")
