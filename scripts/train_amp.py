"""Optional accelerated trainer: identical loss/optimizer to lowlight_train.py,
but the network forward runs under fp16 autocast (losses stay fp32).

Rationale: on this laptop 4060 the fp32 loop only reaches ~35 images/s
(~60 s per 2002-image epoch), so a full 200-epoch run would take hours.
Convs run on tensor cores here; numerics stay close because the curve
estimation is a smooth low-dynamic-range regression.
"""
import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r"\Zero-DCE\Zero-DCE_code")
import dataloader  # noqa: E402
import model  # noqa: E402
import Myloss  # noqa: E402


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lowlight_images_path', type=str, default="data/train_data_sice/")
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--grad_clip_norm', type=float, default=0.1)
    p.add_argument('--num_epochs', type=int, default=200)
    p.add_argument('--train_batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--display_iter', type=int, default=50)
    p.add_argument('--snapshot_iter', type=int, default=10)
    p.add_argument('--snapshots_folder', type=str, default="snapshots_amp/")
    cfg = p.parse_args()

    cudnn.benchmark = True
    os.makedirs(cfg.snapshots_folder, exist_ok=True)

    net = model.enhance_net_nopool().cuda()
    net.apply(weights_init)
    ds = dataloader.lowlight_loader(cfg.lowlight_images_path)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.train_batch_size, shuffle=True,
                                         num_workers=cfg.num_workers, pin_memory=True)
    L_color, L_spa = Myloss.L_color(), Myloss.L_spa()
    L_exp, L_TV = Myloss.L_exp(16, 0.6), Myloss.L_TV()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    net.train()
    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        for it, x in enumerate(loader):
            x = x.cuda(non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, enh, A = net(x)
            enh = enh.float()
            A = A.float()
            loss = (200 * L_TV(A) + torch.mean(L_spa(enh, x)) +
                    5 * torch.mean(L_color(enh)) + 10 * torch.mean(L_exp(enh)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            opt.step()
            if (it + 1) % cfg.display_iter == 0:
                print(f"epoch {epoch} iter {it + 1} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / (epoch * 0 + it + 1):.2f}s/iter avg)", flush=True)
        if (epoch + 1) % cfg.snapshot_iter == 0 or epoch == cfg.num_epochs - 1:
            torch.save(net.state_dict(), os.path.join(cfg.snapshots_folder, f"Epoch{epoch}.pth"))
            print(f"saved Epoch{epoch}.pth  elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
