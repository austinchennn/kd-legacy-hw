"""Teacher (modern ops) and student (legacy-hardware op set) models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ops a typical older edge accelerator / DSP supports natively (INT8 conv pipelines).
LEGACY_OPS = {"Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "AdaptiveAvgPool2d", "Linear", "Flatten",
              "QuantStub", "DeQuantStub", "Sequential", "LegacyCNN", "ConvBNReLU", "Identity"}


# ---------------- Teacher: small ViT (GELU, LayerNorm, softmax attention) ----------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(self, num_classes=10, patch=4, dim=256, depth=6, heads=8):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, patch)
        n = (32 // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        x = self.ln(self.blocks(x)).mean(1)
        return self.head(x)


# ---------------- Student: Conv/BN/ReLU only, INT8-quantizable ----------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class LegacyCNN(nn.Module):
    def __init__(self, num_classes=10, w=(32, 64, 128)):
        super().__init__()
        a, b, c = w
        self.quant = torch.ao.quantization.QuantStub()
        self.features = nn.Sequential(
            ConvBNReLU(3, a), ConvBNReLU(a, a), nn.MaxPool2d(2),
            ConvBNReLU(a, b), ConvBNReLU(b, b), nn.MaxPool2d(2),
            ConvBNReLU(b, c), ConvBNReLU(c, c), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c, num_classes)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.fc(self.features(self.quant(x))))

    def fuse(self):
        for m in self.features:
            if isinstance(m, ConvBNReLU):
                torch.ao.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)


def op_types(model):
    # leaf ops only; attention is reported as one op (its out_proj is internal)
    ops = set()
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            ops.add("MultiheadAttention(softmax)")
        elif not list(m.children()):
            ops.add(type(m).__name__)
    ops.discard("NonDynamicallyQuantizableLinear")
    return sorted(ops)


def unsupported_ops(model):
    return [o for o in op_types(model) if o not in LEGACY_OPS]
