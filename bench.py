"""Evaluate teacher / students under a 'legacy hardware' profile:
op-set compatibility, INT8 post-training quantization accuracy, and
single-thread CPU batch-1 latency (a stand-in for an older edge device).

python bench.py
"""
import copy, glob, json, os, statistics, time
import torch
from models import ViT, LegacyCNN, op_types, unsupported_ops
from train import load_cifar, evaluate

torch.backends.quantized.engine = "qnnpack"
torch.set_num_threads(1)


def latency_ms(model, runs=300, warmup=30):
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def int8(model, calib):
    q = copy.deepcopy(model).eval()
    q.fuse()
    q.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    torch.ao.quantization.prepare(q, inplace=True)
    with torch.inference_mode():
        for k in range(0, len(calib), 256):
            q(calib[k:k + 256])
    return torch.ao.quantization.convert(q, inplace=True)


@torch.no_grad()
def preds(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[k:k + bs]).argmax(1) for k in range(0, len(x), bs)])


def params(m):
    return sum(p.numel() for p in m.parameters())


def size_mb(m):
    torch.save(m.state_dict(), "/tmp/_m.pt")
    s = os.path.getsize("/tmp/_m.pt") / 1e6
    os.remove("/tmp/_m.pt")
    return s


def main():
    data = load_cifar("cpu")
    xte, yte = data["test"]
    calib = data["train"][0][:2048]
    rows = []

    t = ViT()
    t.load_state_dict(torch.load("ckpt/teacher.pt", map_location="cpu"))
    tp = preds(t, xte)
    agree = lambda m: (preds(m, xte) == tp).float().mean().item()
    rows.append({"model": "Teacher ViT (FP32)", "acc": evaluate(t, xte, yte), "agree": 1.0, "params": params(t),
                 "size_mb": size_mb(t), "latency_ms": latency_ms(t.eval()), "unsupported_ops": unsupported_ops(t)})

    for ck in sorted(glob.glob("ckpt/student_*.pt")):
        tag = os.path.basename(ck)[:-3]
        s = LegacyCNN()
        s.load_state_dict(torch.load(ck, map_location="cpu"))
        s.eval()
        q = int8(s, calib)
        base = {"params": params(s), "unsupported_ops": unsupported_ops(s)}
        rows.append({"model": f"{tag} (FP32)", "acc": evaluate(s, xte, yte), "agree": agree(s), "size_mb": size_mb(s),
                     "latency_ms": latency_ms(s), **base})
        rows.append({"model": f"{tag} (INT8)", "acc": evaluate(q, xte, yte), "agree": agree(q), "size_mb": size_mb(q),
                     "latency_ms": latency_ms(q), **base})

    os.makedirs("results", exist_ok=True)
    json.dump({"teacher_ops": op_types(t), "rows": rows}, open("results/bench.json", "w"), indent=1)
    print(f"{'model':34s} {'acc':>7s} {'agree':>7s} {'params':>9s} {'MB':>6s} {'ms(1T,bs1)':>11s}  unsupported ops")
    for r in rows:
        print(f"{r['model']:34s} {r['acc']*100:6.2f}% {r['agree']*100:6.2f}% {r['params']:9d} {r['size_mb']:6.2f} {r['latency_ms']:11.3f}  "
              f"{','.join(r['unsupported_ops']) or '-'}")


if __name__ == "__main__":
    main()
