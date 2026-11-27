"""Evaluate Zero-DCE checkpoints on the LOL pairs (low -> high) with PSNR / SSIM."""
import os, sys, glob, time
import numpy as np
import cv2
import torch

sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model  # noqa: E402

C = r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code"
LOW = os.path.join(C, "data", "train_data")
HIGH = os.path.join(C, "data", "train_high")


def load_rgb(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.transpose())
    f = lambda x: cv2.filter2D(x, -1, win, borderType=cv2.BORDER_REFLECT)
    mu1, mu2 = f(a), f(b)
    mu1s, mu2s = mu1 * mu1, mu2 * mu2
    mu12 = mu1 * mu2
    s1 = f(a * a) - mu1s
    s2 = f(b * b) - mu2s
    s12 = f(a * b) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))
    return float(m.mean())


def enhance(net, img_rgb):
    x = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(x)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def main():
    pairs = sorted(glob.glob(os.path.join(LOW, "*.jpg")))
    pairs = pairs[: int(os.environ.get("N", "0")) or len(pairs)]
    print(f"evaluating {len(pairs)} pairs")
    ckpts = (os.environ.get("CKPTS") or
             "snapshots/Epoch99.pth;snapshots_run/Epoch199.pth;snapshots_run/Epoch99.pth;snapshots_run/Epoch49.pth"
             ).split(";")
    results = {}
    for rel in ckpts:
        ckpt = os.path.join(C, rel)
        if not os.path.exists(ckpt):
            continue
        name = rel.replace("snapshots_run/", "ours_").replace("snapshots/", "official_").replace(".pth", "")
        net = model.enhance_net_nopool().cuda()
        net.load_state_dict(torch.load(ckpt, map_location="cuda"))
        net.eval()
        ps_in = ps_out = ss_in = ss_out = bright_in = bright_out = 0.0
        t0 = time.time()
        for lp in pairs:
            hp = os.path.join(HIGH, os.path.basename(lp))
            lo = load_rgb(lp); hi = load_rgb(hp)
            out = enhance(net, lo)
            ps_in += psnr(lo, hi); ps_out += psnr(out, hi)
            ss_in += ssim(lo, hi); ss_out += ssim(out, hi)
            bright_in += float(lo.mean()); bright_out += float(out.mean())
        n = len(pairs)
        print(f"{name:22s} in PSNR {ps_in/n:6.3f} SSIM {ss_in/n:.4f} | out PSNR {ps_out/n:6.3f} "
              f"SSIM {ss_out/n:.4f} | bright {bright_in/n:.3f}->{bright_out/n:.3f} | {time.time()-t0:.0f}s")
        results[name] = (ps_in / n, ss_in / n, ps_out / n, ss_out / n)
    return results


if __name__ == "__main__":
    main()
