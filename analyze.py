"""Break teacher/student agreement down by teacher correctness (uses saved checkpoints)."""
import glob, json, os
import torch
from models import ViT, LegacyCNN
from train import load_cifar
from bench import preds

xte, yte = load_cifar("cpu")["test"]
t = ViT(); t.load_state_dict(torch.load("ckpt/teacher.pt", map_location="cpu"))
tp = preds(t, xte); tc = tp == yte
out = {"teacher_correct": tc.float().mean().item()}
for ck in sorted(glob.glob("ckpt/student_*.pt")):
    s = LegacyCNN(); s.load_state_dict(torch.load(ck, map_location="cpu"))
    sp = preds(s, xte); sc = sp == yte
    out[os.path.basename(ck)[:-3]] = {
        "agree_when_teacher_right": (sp == tp)[tc].float().mean().item(),
        "agree_when_teacher_wrong": (sp == tp)[~tc].float().mean().item(),
        "student_right_teacher_wrong": (sc & ~tc).float().mean().item(),
        "student_wrong_teacher_right": (~sc & tc).float().mean().item(),
    }
json.dump(out, open("results/agreement_breakdown.json", "w"), indent=1)
print(json.dumps(out, indent=1))
