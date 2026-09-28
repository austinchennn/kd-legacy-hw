"""Accuracy vs single-thread latency plot + training curves from results/*.json."""
import glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

b = json.load(open("results/bench.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for r in b["rows"]:
    c = "tab:red" if "Teacher" in r["model"] else ("tab:blue" if "_kd_" in r["model"] else "tab:gray")
    mk = "s" if "INT8" in r["model"] else "o"
    ax[0].scatter(r["latency_ms"], r["acc"] * 100, c=c, marker=mk)
ax[0].scatter([], [], c="tab:red", label="ViT teacher (FP32)")
ax[0].scatter([], [], c="tab:gray", label="Legacy CNN, scratch")
ax[0].scatter([], [], c="tab:blue", label="Legacy CNN, KD")
ax[0].scatter([], [], c="k", marker="s", label="INT8 (square)")
ax[0].set_xlabel("latency, 1 CPU thread, batch 1 (ms)")
ax[0].set_ylabel("CIFAR-10 test acc (%)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3)
for f in sorted(glob.glob("results/*.json")):
    if "bench" in f:
        continue
    d = json.load(open(f))
    name = f.split("/")[-1][:-5]
    c = "tab:red" if name == "teacher" else ("tab:blue" if "_kd_" in name else "tab:gray")
    ax[1].plot([e["epoch"] for e in d["log"]], [e["test_acc"] * 100 for e in d["log"]], c=c, alpha=.8)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("test acc (%)")
ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig("results/kd_legacy.png", dpi=150)
print("saved results/kd_legacy.png")
