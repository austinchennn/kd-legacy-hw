"""Train teacher (ViT), student from scratch, or student with KD from the teacher.

python train.py --mode teacher --epochs 60
python train.py --mode scratch --epochs 40
python train.py --mode kd --epochs 40 --T 4 --alpha 0.9
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
import torchvision
from models import ViT, LegacyCNN

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar(device):
    out = {}
    for split in (True, False):
        ds = torchvision.datasets.CIFAR10("data", train=split, download=True)
        x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div(255)
        x = (x - MEAN) / STD
        out["train" if split else "test"] = (x.to(device), torch.tensor(ds.targets).to(device))
    return out


def augment(x):
    # random crop (pad 4) + horizontal flip, done on-device
    n = x.size(0)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    x = xp[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for k in range(0, len(x), bs):
        correct += (model(x[k:k + bs]).argmax(1) == y[k:k + bs]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "scratch", "kd"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_cifar(dev)
    xtr, ytr = data["train"]
    xte, yte = data["test"]

    if args.mode == "teacher":
        model = ViT().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    else:
        model = LegacyCNN().to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    teacher = None
    if args.mode == "kd":
        teacher = ViT().to(dev)
        teacher.load_state_dict(torch.load("ckpt/teacher.pt", map_location=dev))
        teacher.eval()

    steps = args.epochs * (len(xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.15)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for k in range(len(xtr) // args.bs):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            x, y = augment(xtr[idx]), ytr[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1 if args.mode == "teacher" else 0.0)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(x)
                kd = F.kl_div(F.log_softmax(logits / args.T, 1), F.softmax(tl / args.T, 1), reduction="batchmean") * args.T ** 2
                loss = args.alpha * kd + (1 - args.alpha) * F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate(model, xte, yte)
        log.append({"epoch": ep + 1, "test_acc": acc, "time_s": round(time.time() - t0, 1)})
        print(f"[{args.mode}] ep {ep + 1}/{args.epochs} acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    name = args.mode if args.mode == "teacher" else f"student_{args.mode}_s{args.seed}"
    torch.save(model.state_dict(), f"ckpt/{name}.pt")
    json.dump({"args": vars(args), "log": log, "final_acc": log[-1]["test_acc"]}, open(f"results/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
