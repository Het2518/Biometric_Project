import json, textwrap

def code(src): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src}
def md(src):   return {'cell_type':'markdown','metadata':{},'source':src}

cells = []

# ─── CELL 0 ───────────────────────────────────────────────────────────────────
cells.append(md("""\
# Adversarial Identity Leakage in De-Identification — GPU-First Pipeline
### Paper: Rosberg et al., IEEE TBIOM Vol.8 No.2, March 2026

**Novelty added — *Learned Spectral Defense (LSD)*:**  
Instead of the paper's fixed Gaussian blur, a small trainable CNN learns a *soft mask in FFT space*
that adaptively suppresses adversarial frequency components while preserving identity information.

| Stage | Description |
|---|---|
| 1 | Install / Imports / Config |
| 2 | Identity encoders (ArcFace victim + CosFace surrogate) |
| 3 | Adversarial attacks: PGD L∞/L2, Frank-Wolfe, Elastic, Fog, Snow |
| 4 | Learned Attack U-Net (ReFace style) — **load or train** |
| 5 | FIVA de-identification pipeline |
| 6 | Robust ArcFace fine-tuning (knowledge distillation) — **load or train** |
| 7 | Defense A: Gaussian low-pass filter (paper baseline) |
| 8 | **Defense B: Learned Spectral Defense / LSD (NOVELTY) — load or train** |
| 9 | Evaluation: FAR-based identity leakage tables |
|10 | Plots + qualitative visualisation |
"""))

# ─── CELL 1: Install ──────────────────────────────────────────────────────────
cells.append(md("## Cell 1 — Install"))
cells.append(code("""\
import subprocess, sys
pkgs = ["facenet-pytorch", "kornia", "timm", "insightface", "onnxruntime-gpu", "tqdm"]
for p in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", p, "-q"], capture_output=True)
print("Packages ready.")
"""))

# ─── CELL 2: Imports ──────────────────────────────────────────────────────────
cells.append(md("## Cell 2 — Imports"))
cells.append(code("""\
import os, copy, math, random, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch.amp import autocast, GradScaler
import kornia

warnings.filterwarnings("ignore")

# ── GPU tuning ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark      = True   # fastest conv kernel for fixed size
torch.backends.cudnn.deterministic  = False  # allow non-determinism for speed

print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"torch  : {torch.__version__}")
"""))

# ─── CELL 3: Config ───────────────────────────────────────────────────────────
cells.append(md("## Cell 3 — Config  *(edit paths then re-run once)*"))
cells.append(code("""\
# ── EDIT THESE PATHS ─────────────────────────────────────────────────────────
DATA_ROOT   = "./data"
FF_ROOT     = f"E:\FaceForensics++_C23\original"             # FF++ pre-extracted frames
CELEBA_ROOT = f""E:\img_align_celeba\img_align_celeba""      # CelebA aligned imgs
LFW_ROOT    = f"E:\LFW\lfw-deepfunneled"  # LFW  person sub-dirs
CKPT_DIR    = "./checkpoints"
OUT_DIR     = "./outputs"
# ─────────────────────────────────────────────────────────────────────────────

for d in [CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyper-parameters (exact paper values) ────────────────────────────────────
CFG = dict(
    img_size          = 112,     # ArcFace standard
    batch_size        = 64,      # lower if OOM (32 / 16)
    num_workers       = 4,
    pin_memory        = True,
    lr                = 1e-4,    # AdamW lr  (paper)
    betas             = (0.9, 0.999),
    n_epochs_unet     = 10,      # learned attack U-Net
    n_epochs_finetune = 10,      # robust ArcFace + LSD
    lam_rob           = 1,       # Eq.1 coefficients (paper)
    lam_con           = 3,
    lam_union         = 2,
    lam_tv            = 1e-4,    # TV loss for learned attack
    eps_p             = 0.05,    # learned attack strength
    fiva_margin       = 0.3,     # ITM margin (paper)
    fiva_track        = False,   # tracking off = fresh fake per frame (paper eval)

    # Table I attack budgets
    attacks = {
        "pgd_linf"   : dict(eps=16/255,   n=50, norm="linf"),
        "pgd_l2"     : dict(eps=2400/255, n=50, norm="l2"),
        "frank_wolfe": dict(eps=13/255,   n=50),
        "elastic"    : dict(eps=8,        n=50),
        "fog"        : dict(eps=600,      n=50),
        "snow"       : dict(eps=0.125,    n=50),
    },
    gauss_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0],  # Figure 5
    far_values   = [1e-3, 1e-4, 1e-5],           # Table II

    # Subset sizes — set None for full dataset
    ff_max_imgs  = 500,
    celeba_max   = 20000,
)
print("Config ready.")
print(f"  batch={CFG['batch_size']}  img={CFG['img_size']}  AMP=True  cudnn.benchmark=True")
"""))

# ─── CELL 4: Datasets ─────────────────────────────────────────────────────────
cells.append(md("## Cell 4 — Datasets"))
cells.append(code("""\
face_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),   # -> [-1, 1]
])

class FlatImageDataset(Dataset):
    \"\"\"Recursively finds all jpg/png in root, optional cap.\"\"\"
    def __init__(self, root, max_imgs=None, transform=None):
        root  = Path(root)
        paths = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
        self.paths = paths[:max_imgs] if max_imgs else paths
        self.tf    = transform
        print(f"  {root.name}: {len(self.paths)} images")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return (self.tf(img) if self.tf else img), str(self.paths[i])

class LFWDataset(Dataset):
    \"\"\"LFW with person sub-dirs → (img, person_id).\"\"\"
    def __init__(self, root, transform=None):
        root = Path(root)
        self.samples, self.id2idx = [], {}
        for idx, pdir in enumerate(sorted(root.iterdir())):
            if not pdir.is_dir(): continue
            self.id2idx[pdir.name] = idx
            for f in sorted(pdir.glob("*.jpg")):
                self.samples.append((str(f), idx))
        self.tf = transform
        print(f"  LFW: {len(self.samples)} imgs, {len(self.id2idx)} identities")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, lbl = self.samples[i]
        img = Image.open(p).convert("RGB")
        return (self.tf(img) if self.tf else img), lbl

print("Loading datasets...")
ff_ds  = FlatImageDataset(FF_ROOT,     CFG["ff_max_imgs"], face_tf)
cel_ds = FlatImageDataset(CELEBA_ROOT, CFG["celeba_max"],  face_tf)
lfw_ds = LFWDataset(LFW_ROOT, face_tf)

def make_loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
                      persistent_workers=True, prefetch_factor=2)

ff_loader  = make_loader(ff_ds,  shuffle=False)
cel_loader = make_loader(cel_ds, shuffle=True)
lfw_loader = make_loader(lfw_ds, shuffle=False)
print("Loaders ready.")
"""))

# ─── CELL 5: Identity encoders ────────────────────────────────────────────────
cells.append(md("## Cell 5 — Identity Encoders"))
cells.append(code("""\
import timm
from facenet_pytorch import InceptionResnetV1

# ── Shared IResNet-style backbone ─────────────────────────────────────────────
def build_iresnet50():
    \"\"\"ResNet50 + 512-d BN head — ArcFace-family backbone approximation.\"\"\"
    m        = timm.create_model("resnet50", pretrained=True, num_classes=0)
    feat_dim = m.num_features
    m.head   = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim, 512), nn.BatchNorm1d(512))
    return m

class IdentityEncoder(nn.Module):
    \"\"\"Unified wrapper — forward() returns L2-normalised 512-d embedding.\"\"\"
    def __init__(self, name, backbone, metric="cosine"):
        super().__init__()
        self.name     = name
        self.backbone = backbone
        self.metric   = metric       # "cosine" | "l2" | "undefined"
    def forward(self, x):
        emb = self.backbone(x)
        if isinstance(emb, (list, tuple)): emb = emb[0]
        return F.normalize(emb, p=2, dim=1)

# ── Distance helpers ──────────────────────────────────────────────────────────
def cosine_dist(a, b): return (1 - F.cosine_similarity(a, b, dim=1)).mean()
def l2_dist(a, b):     return torch.norm(a - b, p=2, dim=1).mean()
def loss_for(enc):     return l2_dist if enc.metric == "l2" else cosine_dist

# ── Build all encoders ────────────────────────────────────────────────────────
encoders = {}

# ArcFace (victim)
encoders["arcface"] = IdentityEncoder("ArcFace", build_iresnet50(), "cosine").to(device).eval()

# CosFace (strongest surrogate — cosine metric space, same as victim)
encoders["cosface"] = IdentityEncoder("CosFace", copy.deepcopy(encoders["arcface"].backbone),
                                       "cosine").to(device).eval()

# FaceNet (L2 metric space — different from victim)
fn = InceptionResnetV1(pretrained="vggface2").eval()
encoders["facenet"] = IdentityEncoder("FaceNet", fn, "l2").to(device).eval()

# ResNet50-ImageNet (undefined space — weakest expected surrogate)
rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
rn.fc = nn.Identity()
encoders["resnet50"] = IdentityEncoder("ResNet50-IN", rn, "undefined").to(device).eval()

# ── torch.compile (PyTorch >= 2.0) ───────────────────────────────────────────
try:
    for k in list(encoders.keys()):
        encoders[k] = torch.compile(encoders[k], mode="reduce-overhead")
    print("torch.compile applied -> faster inference")
except Exception as e:
    print(f"torch.compile skipped ({e})")

print(f"Encoders ready: {list(encoders.keys())}")
"""))

# ─── CELL 6: Attacks ──────────────────────────────────────────────────────────
cells.append(md("## Cell 6 — Adversarial Attacks (GPU, paper Table I params)"))
cells.append(code("""\
# All attacks operate on GPU tensors in [-1, 1].
# Objective: MAXIMISE cosine (or L2) distance of embeddings.

def pgd(x, enc, eps, n=50, norm="linf"):
    \"\"\"PGD in L-inf or L2 ball.  All-GPU, no Python loop overhead via in-place.\"\"\"
    lf   = loss_for(enc)
    step = eps * 2 / n
    x0   = x.detach()
    xadv = x0.clone()
    with torch.no_grad():
        e0 = enc(x0)
    for _ in range(n):
        xadv = xadv.detach().requires_grad_(True)
        loss = -lf(e0, enc(xadv))
        g    = torch.autograd.grad(loss, xadv)[0]
        with torch.no_grad():
            if norm == "linf":
                delta = (xadv - step * g.sign() - x0).clamp(-eps, eps)
            else:   # l2
                gn    = g.view(g.shape[0], -1).norm(p=2, dim=1).view(-1,1,1,1).clamp(1e-8)
                delta = xadv - step * (g / gn) - x0
                dn    = delta.view(delta.shape[0],-1).norm(p=2,dim=1).view(-1,1,1,1).clamp(1e-8)
                delta = delta * (eps / dn).clamp(max=1)
            xadv = (x0 + delta).clamp(-1, 1)
    return xadv.detach()


def frank_wolfe(x, enc, eps=13/255, n=50):
    \"\"\"Conditional gradient (Frank-Wolfe) in L-inf ball.\"\"\"
    lf    = loss_for(enc)
    x0    = x.detach()
    delta = torch.zeros_like(x0)
    with torch.no_grad():
        e0 = enc(x0)
    for t in range(1, n + 1):
        xadv = (x0 + delta).clamp(-1, 1).requires_grad_(True)
        g    = torch.autograd.grad(-lf(e0, enc(xadv)), xadv)[0]
        with torch.no_grad():
            s     = -eps * g.sign()
            gamma = 2.0 / (t + 2)
            delta = ((1 - gamma) * delta + gamma * (s - x0)).clamp(-eps, eps)
    return (x0 + delta).clamp(-1, 1).detach()


def elastic(x, enc, eps=8, n=50):
    \"\"\"Adversarial elastic warp — max displacement = eps pixels.\"\"\"
    lf      = loss_for(enc)
    B,C,H,W = x.shape
    x0      = x.detach()
    with torch.no_grad():
        e0 = enc(x0)
    flow = torch.zeros(B, H, W, 2, device=device, requires_grad=True)
    opt  = torch.optim.Adam([flow], lr=eps / n * 2)
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device),
                             torch.linspace(-1,1,W,device=device), indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0).expand(B,-1,-1,-1)
    for _ in range(n):
        grid   = base + flow.clamp(-eps/H, eps/H)
        warped = F.grid_sample(x0, grid, align_corners=True, padding_mode="border")
        loss   = lf(e0, enc(warped))
        opt.zero_grad(); loss.backward()
        flow.grad.mul_(-1); opt.step()    # gradient ascent
    with torch.no_grad():
        grid   = base + flow.clamp(-eps/H, eps/H)
        warped = F.grid_sample(x0, grid, align_corners=True, padding_mode="border")
    return warped.detach()


def fog(x, enc, eps=600, n=50):
    \"\"\"Adversarial smooth fog overlay.\"\"\"
    lf      = loss_for(enc)
    B,C,H,W = x.shape
    x0      = x.detach()
    with torch.no_grad():
        e0 = enc(x0)
    fog_base = F.interpolate(torch.rand(B,1,H//8,W//8,device=device),
                              (H,W), mode="bilinear", align_corners=True).expand(B,C,H,W).clone()
    s   = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([s], lr=0.01)
    mx  = eps / 255 / 2
    for _ in range(n):
        xf   = (x0 + fog_base * torch.sigmoid(s) * mx).clamp(-1,1)
        loss = lf(e0, enc(xf))
        opt.zero_grad(); loss.backward()
        s.grad.mul_(-1); opt.step()
    with torch.no_grad():
        xf = (x0 + fog_base * torch.sigmoid(s) * mx).clamp(-1,1)
    return xf.detach()


def snow(x, enc, eps=0.125, n=50):
    \"\"\"Adversarial sparse bright-pixel (snow) perturbation.\"\"\"
    lf   = loss_for(enc)
    x0   = x.detach()
    with torch.no_grad():
        e0 = enc(x0)
    mask = torch.zeros_like(x0, requires_grad=True)
    opt  = torch.optim.Adam([mask], lr=0.01)
    for _ in range(n):
        xs   = (x0 + torch.sigmoid(mask) * eps).clamp(-1,1)
        loss = lf(e0, enc(xs))
        opt.zero_grad(); loss.backward()
        mask.grad.mul_(-1); opt.step()
    with torch.no_grad():
        xs = (x0 + torch.sigmoid(mask) * eps).clamp(-1,1)
    return xs.detach()


# ── Master dispatcher ─────────────────────────────────────────────────────────
ATTACKS = {
    "pgd_linf"   : lambda x,e: pgd(x, e, eps=CFG["attacks"]["pgd_linf"]["eps"],   norm="linf"),
    "pgd_l2"     : lambda x,e: pgd(x, e, eps=CFG["attacks"]["pgd_l2"]["eps"],     norm="l2"),
    "frank_wolfe": lambda x,e: frank_wolfe(x, e, eps=CFG["attacks"]["frank_wolfe"]["eps"]),
    "elastic"    : lambda x,e: elastic(x, e, eps=CFG["attacks"]["elastic"]["eps"]),
    "fog"        : lambda x,e: fog(x, e,   eps=CFG["attacks"]["fog"]["eps"]),
    "snow"       : lambda x,e: snow(x, e,  eps=CFG["attacks"]["snow"]["eps"]),
}
print(f"Attacks: {list(ATTACKS.keys())}")
"""))

# ─── CELL 7: Learned Attack U-Net ─────────────────────────────────────────────
cells.append(md("## Cell 7 — Learned Attack U-Net  *(load or train)*"))
cells.append(code("""\
# ReFace-style U-Net: learns universal adversarial perturbation.
# Trained on CelebA with cosine-distance maximisation + TV regularisation.

class UBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1), nn.InstanceNorm2d(oc), nn.LeakyReLU(.2,True),
            nn.Conv2d(oc,oc,3,padding=1), nn.InstanceNorm2d(oc), nn.LeakyReLU(.2,True))
    def forward(self,x): return self.net(x)

class LearnedAttackNet(nn.Module):
    \"\"\"Outputs perturbation in [-1,1]; applied as x_adv = clamp(x + p * eps_p).\"\"\"
    def __init__(self):
        super().__init__()
        self.e1 = UBlock(3,32);   self.e2 = UBlock(32,64)
        self.e3 = UBlock(64,128); self.bn = UBlock(128,256)
        self.d3 = UBlock(256+128,128); self.d2 = UBlock(128+64,64)
        self.d1 = UBlock(64+32,32);    self.out = nn.Conv2d(32,3,1)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2)); b=self.bn(self.pool(e3))
        d=self.d3(torch.cat([self.up(b),e3],1))
        d=self.d2(torch.cat([self.up(d),e2],1))
        d=self.d1(torch.cat([self.up(d),e1],1))
        return torch.tanh(self.out(d))

def tv_loss(x):
    return (torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]).mean() +
            torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).mean())

UNET_CKPT = f"{CKPT_DIR}/learned_attack_unet.pt"
attack_unet = LearnedAttackNet().to(device)

if os.path.exists(UNET_CKPT):
    # ── LOAD ─────────────────────────────────────────────────────────────────
    attack_unet.load_state_dict(torch.load(UNET_CKPT, map_location=device)["model"])
    attack_unet.eval()
    print(f"[LOAD] Learned attack U-Net from {UNET_CKPT}")
else:
    # ── TRAIN ────────────────────────────────────────────────────────────────
    print("Training learned attack U-Net...")
    victim = encoders["arcface"]
    opt    = torch.optim.AdamW(attack_unet.parameters(), lr=CFG["lr"], betas=CFG["betas"])
    scaler = GradScaler()
    eps_p  = CFG["eps_p"]
    attack_unet.train()
    for ep in range(CFG["n_epochs_unet"]):
        tloss = 0.0; nb = 0
        for imgs, _ in tqdm(cel_loader, desc=f"U-Net Ep {ep+1}/{CFG['n_epochs_unet']}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                p     = attack_unet(imgs)
                xadv  = (imgs + p * eps_p).clamp(-1, 1)
                ec    = victim(imgs).detach()
                ea    = victim(xadv)
                loss  = -cosine_dist(ec, ea) + CFG["lam_tv"] * tv_loss(p)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tloss += loss.item(); nb += 1
        print(f"  Ep {ep+1}  loss={tloss/nb:.4f}")
    attack_unet.eval()
    torch.save({"model": attack_unet.state_dict()}, UNET_CKPT)
    print(f"[SAVE] {UNET_CKPT}")

def apply_learned_attack(x, eps_p=None):
    eps_p = eps_p or CFG["eps_p"]
    with torch.no_grad():
        p = attack_unet(x)
    return (x + p * eps_p).clamp(-1, 1)
"""))

# ─── CELL 8: FIVA ─────────────────────────────────────────────────────────────
cells.append(md("## Cell 8 — FIVA De-identification Pipeline"))
cells.append(code("""\
# Lightweight FIVA: ITM samples fake identity, AdaIN UNet swaps face.

class AdaIN(nn.Module):
    \"\"\"Adaptive Instance Norm — inject fake identity into feature maps.\"\"\"
    def __init__(self, ch, id_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.proj = nn.Linear(id_dim, ch * 2)
    def forward(self, x, z):
        g, b = self.proj(z).chunk(2, dim=1)
        return self.norm(x) * (1 + g.view(*g.shape,1,1)) + b.view(*b.shape,1,1)

class FIVAGenerator(nn.Module):
    def __init__(self, base=64, id_dim=512):
        super().__init__()
        self.e1  = UBlock(3,base);        self.e2  = UBlock(base,   base*2)
        self.e3  = UBlock(base*2,base*4); self.bot = UBlock(base*4, base*8)
        self.a3  = AdaIN(base*8,  id_dim); self.a2 = AdaIN(base*4, id_dim)
        self.a1  = AdaIN(base*2,  id_dim)
        self.d3  = UBlock(base*12,base*4); self.d2 = UBlock(base*6,  base*2)
        self.d1  = UBlock(base*3, base);   self.out = nn.Conv2d(base, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    def forward(self, x, z):
        e1=self.e1(x); e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2)); b=self.a3(self.bot(self.pool(e3)),z)
        d=self.d3(torch.cat([self.up(b),e3],1)); d=self.a2(d,z)
        d=self.d2(torch.cat([self.up(d),e2],1)); d=self.a1(d,z)
        d=self.d1(torch.cat([self.up(d),e1],1))
        return torch.tanh(self.out(d))

def sample_fake_id(real_emb, margin=0.3, tries=100):
    \"\"\"Sample unit vector on hypersphere with cos-distance >= margin.\"\"\"
    B, D = real_emb.shape
    for _ in range(tries):
        c = F.normalize(torch.randn(B, D, device=device), p=2, dim=1)
        if ((1 - F.cosine_similarity(real_emb, c, dim=1)) >= margin).all():
            return c
    return F.normalize(-real_emb + 0.1*torch.randn_like(real_emb), p=2, dim=1)

class FIVA(nn.Module):
    def __init__(self, encoder, margin=0.3):
        super().__init__()
        self.enc    = encoder
        self.gen    = FIVAGenerator()
        self.margin = margin
    def forward(self, x, gauss_sigma=0.0):
        x_id = x
        if gauss_sigma > 0:
            k    = int(gauss_sigma * 6) | 1
            x_id = kornia.filters.gaussian_blur2d(x, (k,k), (gauss_sigma,gauss_sigma))
        with torch.no_grad():
            rz = self.enc(x_id)
            fz = sample_fake_id(rz, self.margin)
        return self.gen(x, fz), rz, fz

fiva = FIVA(encoders["arcface"], margin=CFG["fiva_margin"]).to(device)
print(f"FIVA ready — generator params: {sum(p.numel() for p in fiva.gen.parameters())//1000} K")
"""))

# ─── CELL 9: Robust ArcFace ────────────────────────────────────────────────────
cells.append(md("## Cell 9 — Robust ArcFace Fine-tuning  *(load or train)*"))
cells.append(code("""\
# Knowledge distillation (paper Eq.1).
# Student has dual BN heads: one for clean, one for adversarial input.

class RobustArcFace(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        import timm
        # Share the ResNet50 body, duplicate only the final projection
        m = timm.create_model("resnet50", pretrained=False, num_classes=0)
        m.load_state_dict(base_encoder.backbone.state_dict(), strict=False)
        self.body       = m
        feat             = m.num_features
        self.head_clean = nn.Sequential(nn.Flatten(), nn.Linear(feat,512), nn.BatchNorm1d(512))
        self.head_adv   = nn.Sequential(nn.Flatten(), nn.Linear(feat,512), nn.BatchNorm1d(512))
        # Initialise heads from original weights
        orig_head = base_encoder.backbone.head
        for h in [self.head_clean, self.head_adv]:
            h[1].weight.data.copy_(orig_head[1].weight.data)
            h[2].weight.data.copy_(orig_head[2].weight.data)
    def forward(self, x, adv=False):
        feat = self.body(x)
        h    = self.head_adv if adv else self.head_clean
        return F.normalize(h(feat), p=2, dim=1)

ROB_CKPT = f"{CKPT_DIR}/robust_arcface.pt"

if os.path.exists(ROB_CKPT):
    # ── LOAD ─────────────────────────────────────────────────────────────────
    rob_arc = RobustArcFace(encoders["arcface"]).to(device)
    rob_arc.load_state_dict(torch.load(ROB_CKPT, map_location=device))
    rob_arc.eval()
    print(f"[LOAD] Robust ArcFace from {ROB_CKPT}")
else:
    # ── TRAIN ────────────────────────────────────────────────────────────────
    print("Fine-tuning Robust ArcFace (paper Eq.1)...")
    teacher = encoders["arcface"]
    rob_arc = RobustArcFace(teacher).to(device)
    opt     = torch.optim.AdamW(rob_arc.parameters(), lr=CFG["lr"], betas=CFG["betas"])
    scaler  = GradScaler()
    lr, lc, lu = CFG["lam_rob"], CFG["lam_con"], CFG["lam_union"]
    rob_arc.train()
    for ep in range(CFG["n_epochs_finetune"]):
        tloss = 0.0; nb = 0
        for imgs, _ in tqdm(cel_loader, desc=f"Rob Ep {ep+1}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            xadv = apply_learned_attack(imgs, eps_p=random.uniform(0.01, CFG["eps_p"]))
            if random.random() > 0.5:
                xadv = ATTACKS[random.choice(["pgd_linf","fog","snow"])](xadv, encoders["arcface"])
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                ztc = teacher(imgs).detach()        # clean teacher
                zsc = rob_arc(imgs,  adv=False)     # student clean
                zsa = rob_arc(xadv, adv=True)       # student adversarial
                L   = lr*cosine_dist(ztc,zsa) + lc*cosine_dist(ztc,zsc) + lu*cosine_dist(zsc,zsa)
            scaler.scale(L).backward(); scaler.step(opt); scaler.update()
            tloss += L.item(); nb += 1
        print(f"  Ep {ep+1}  loss={tloss/nb:.4f}")
    rob_arc.eval()
    torch.save(rob_arc.state_dict(), ROB_CKPT)
    print(f"[SAVE] {ROB_CKPT}")

# Wrap in IdentityEncoder API so the pipeline is uniform
class RobEncWrapper(nn.Module):
    metric = "cosine"
    def __init__(self, m): super().__init__(); self.backbone = m
    def forward(self, x): return self.backbone(x, adv=False)

encoders["arcface_robust"] = RobEncWrapper(rob_arc).to(device).eval()
# Also create FIVA with robust encoder
fiva_robust = FIVA(encoders["arcface_robust"], margin=CFG["fiva_margin"]).to(device).eval()
print("Robust FIVA ready.")
"""))

# ─── CELL 10: Learned Spectral Defense (NOVELTY) ───────────────────────────────
cells.append(md("""\
## Cell 10 — 🆕 NOVELTY: Learned Spectral Defense (LSD)  *(load or train)*

**Key idea:** The paper uses a *fixed* Gaussian blur to suppress high-frequency adversarial noise.
We replace it with a **trainable CNN-based frequency-domain mask** that:

1. Computes the 2-D real FFT of each input channel.  
2. A 3-layer CNN predicts a **soft mask in [0,1]** over the magnitude spectrum.  
3. The mask is applied to the spectrum, then inverse FFT reconstructs the image.  
4. Trained end-to-end to (a) push adversarial embeddings back toward the clean embedding,
   and (b) be a near-identity mapping on clean images (reconstruction loss).

**Why this is better than Gaussian blur:**  
- *Adaptive*: learns which frequency bands carry adversarial energy for each attack type.  
- *Universal*: a single model handles PGD (high-freq noise) and distortion attacks (fog/snow).  
- *Preserves identity*: avoids over-smoothing mid-frequency identity cues that Gaussian blurs.
"""))
cells.append(code("""\
class LearnedSpectralDefense(nn.Module):
    \"\"\"
    Learnable frequency-domain filter.
    Input/output: [B, 3, H, W] in [-1, 1].

    Architecture:
      per-channel log-magnitude spectrum → 3-layer CNN → soft mask in [0,1]
      → element-wise mask on complex spectrum → iFFT → clamp
    \"\"\"
    def __init__(self, img_size=112):
        super().__init__()
        # mask_net operates on log-magnitude of rfft2 output: [B, 3, H, W//2+1]
        self.mask_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16,16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()                    # mask values in [0, 1]
        )
        # Init bias → mask ≈ 1 (pass-through) at start
        nn.init.constant_(self.mask_net[-2].bias, 5.0)

    def forward(self, x):
        # 1. FFT  (rfft2 = real FFT, output shape [B, C, H, W//2+1] complex)
        X     = torch.fft.rfft2(x, norm="ortho")
        mag   = X.abs().clamp(min=1e-8)      # magnitude
        phase = X / mag                       # unit-phase (direction)

        # 2. Predict spectral mask from log-magnitude
        log_mag = torch.log(mag)
        mask    = self.mask_net(log_mag)      # [B, 3, H, W//2+1]

        # 3. Apply mask and reconstruct via iFFT
        X_filt = mask * mag * phase
        x_rec  = torch.fft.irfft2(X_filt, s=(x.shape[2], x.shape[3]), norm="ortho")
        return x_rec.clamp(-1, 1)


LSD_CKPT = f"{CKPT_DIR}/learned_spectral_defense.pt"
lsd = LearnedSpectralDefense(CFG["img_size"]).to(device)

if os.path.exists(LSD_CKPT):
    # ── LOAD ─────────────────────────────────────────────────────────────────
    lsd.load_state_dict(torch.load(LSD_CKPT, map_location=device))
    lsd.eval()
    print(f"[LOAD] LSD from {LSD_CKPT}")
else:
    # ── TRAIN ────────────────────────────────────────────────────────────────
    # Objective:
    #   L_adv = cosine_dist(victim(lsd(x_adv)), victim(x_clean))  <- fix leakage
    #   L_rec = MSE(lsd(x_clean), x_clean)                        <- identity on clean
    print("Training Learned Spectral Defense...")
    victim = encoders["arcface"]
    opt    = torch.optim.AdamW(lsd.parameters(), lr=CFG["lr"]*2, betas=CFG["betas"])
    scaler = GradScaler()
    lsd.train()
    for ep in range(CFG["n_epochs_finetune"]):
        tloss = 0.0; nb = 0
        for imgs, _ in tqdm(cel_loader, desc=f"LSD Ep {ep+1}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            # Diverse attacks via CosFace surrogate (black-box setting)
            atk  = random.choice(list(ATTACKS.keys()))
            xadv = ATTACKS[atk](imgs, encoders["cosface"])

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                x_def  = lsd(xadv)          # defended adversarial image
                x_pass = lsd(imgs)          # LSD on clean (should be near-identity)
                # Goal 1: adversarial embedding → clean embedding
                zc    = victim(imgs).detach()
                zd    = victim(x_def)
                L_adv = cosine_dist(zc, zd)
                # Goal 2: preserve clean images
                L_rec = F.mse_loss(x_pass, imgs)
                loss  = L_adv + 0.5 * L_rec

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tloss += loss.item(); nb += 1
        print(f"  Ep {ep+1}  loss={tloss/nb:.4f}")
    lsd.eval()
    torch.save(lsd.state_dict(), LSD_CKPT)
    print(f"[SAVE] {LSD_CKPT}")

print(f"LSD params: {sum(p.numel() for p in lsd.parameters())//1000} K  (very lightweight)")
lsd_fn = lambda x: lsd(x)      # convenience wrapper for eval loop
"""))

# ─── CELL 11: FAR calibration ───────────────────────────────────────────────────
cells.append(md("## Cell 11 — FAR Threshold Calibration"))
cells.append(code("""\
@torch.no_grad()
def compute_far_thresholds(encoder, loader, far_targets, max_batches=20):
    \"\"\"
    Compute score thresholds at given FAR targets using the CelebA loader.
    Genuine  = consecutive images (approximation).
    Impostor = half-dataset-offset pairs (random impostor).
    \"\"\"
    encoder.eval()
    embs = []
    for i, (imgs, _) in enumerate(loader):
        if i >= max_batches: break
        with autocast(device_type="cuda"):
            e = encoder(imgs.to(device, non_blocking=True))
        embs.append(e.float().cpu())
    embs = torch.cat(embs, 0)
    N    = embs.shape[0]

    gen_sims = F.cosine_similarity(embs[:-1], embs[1:], dim=1).numpy()
    idx_a    = torch.arange(N)
    idx_b    = (idx_a + N // 2) % N
    imp_sims = F.cosine_similarity(embs[idx_a], embs[idx_b], dim=1).numpy()

    thresholds = {}
    for far in far_targets:
        thresholds[far] = float(np.quantile(imp_sims, 1 - far))
    return thresholds

print("Calibrating FAR thresholds...")
far_thresholds = compute_far_thresholds(
    encoders["arcface"], cel_loader, CFG["far_values"], max_batches=20)
for f, t in far_thresholds.items():
    print(f"  FAR {f:.0e}  =>  threshold = {t:.4f}")
"""))

# ─── CELL 12: Leakage measurement ─────────────────────────────────────────────
cells.append(md("## Cell 12 — Identity Leakage Measurement Function"))
cells.append(code("""\
@torch.no_grad()
def measure_leakage(attack_fn, surrogate_enc, eval_enc, fiva_sys,
                    loader, thresholds, defense=None, max_batches=15):
    \"\"\"
    Returns {far: TAR} — TAR = identity leakage rate (lower = better de-id).
    Pipeline: attack -> [defense] -> FIVA -> eval identity match.
    \"\"\"
    eval_enc.eval(); fiva_sys.eval()
    hits  = {f: 0 for f in thresholds}
    total = 0
    for i, (imgs, _) in enumerate(loader):
        if i >= max_batches: break
        imgs = imgs.to(device, non_blocking=True)
        # 1. Attack
        xadv = attack_fn(imgs, surrogate_enc)
        # 2. Defense (optional)
        if defense is not None:
            xadv = defense(xadv)
        # 3. De-identify
        deid, _, _ = fiva_sys(xadv)
        # 4. Check leakage: does eval_enc match original identity?
        z_orig = eval_enc(imgs).float()
        z_deid = eval_enc(deid).float()
        sims   = F.cosine_similarity(z_orig, z_deid, dim=1).cpu().numpy()
        for f, thr in thresholds.items():
            hits[f] += int((sims >= thr).sum())
        total += imgs.shape[0]
    return {f: hits[f] / max(total, 1) for f in thresholds}
"""))

# ─── CELL 13: Full evaluation table ────────────────────────────────────────────
cells.append(md("## Cell 13 — Full Evaluation (Table II equivalent)"))
cells.append(code("""\
# ── Defense functions ─────────────────────────────────────────────────────────
gauss_15 = lambda x: kornia.filters.gaussian_blur2d(x, (9,9), (1.5,1.5))  # paper sigma=1.5

MAX_BATCHES = 10   # increase for full-scale; 10 batches = fast smoke-test

configs = [
    # (attack,        surrogate,   defense_label,  defense_fn,  fiva_sys)
    ("pgd_linf",    "cosface",  "none",     None,     fiva),
    ("pgd_linf",    "cosface",  "gaussian", gauss_15, fiva),
    ("pgd_linf",    "cosface",  "LSD",      lsd_fn,   fiva),
    ("pgd_linf",    "cosface",  "none",     None,     fiva_robust),
    ("frank_wolfe", "cosface",  "none",     None,     fiva),
    ("frank_wolfe", "cosface",  "gaussian", gauss_15, fiva),
    ("frank_wolfe", "cosface",  "LSD",      lsd_fn,   fiva),
    ("fog",         "cosface",  "none",     None,     fiva),
    ("fog",         "cosface",  "gaussian", gauss_15, fiva),
    ("fog",         "cosface",  "LSD",      lsd_fn,   fiva),
    ("snow",        "cosface",  "none",     None,     fiva),
    ("snow",        "cosface",  "LSD",      lsd_fn,   fiva),
    ("elastic",     "facenet",  "none",     None,     fiva),   # L2 metric surrogate
    ("elastic",     "facenet",  "LSD",      lsd_fn,   fiva),
]

RESULTS = {}
for atk, sur, dname, dfn, fsys in tqdm(configs, desc="Evaluating"):
    key = f"{atk}|{sur}|{dname}|{'robust' if fsys is fiva_robust else 'orig'}"
    tar = measure_leakage(ATTACKS[atk], encoders[sur], encoders["arcface"],
                          fsys, ff_loader, far_thresholds,
                          defense=dfn, max_batches=MAX_BATCHES)
    RESULTS[key] = tar
    short = {f"FAR{f:.0e}": f"{v:.4f}" for f,v in tar.items()}
    print(f"  {key:55s}  {short}")

print("\\nEvaluation complete.")
"""))

# ─── CELL 14: Results table ────────────────────────────────────────────────────
cells.append(md("## Cell 14 — Results Table"))
cells.append(code("""\
import pandas as pd

rows = []
for key, tar in RESULTS.items():
    atk, sur, dname, fv = key.split("|")
    row = {"attack": atk, "surrogate": sur, "defense": dname, "fiva": fv}
    for f, v in tar.items():
        row[f"TAR@FAR{f:.0e}"] = round(v, 4)
    rows.append(row)

df = pd.DataFrame(rows).sort_values(["attack","defense"])
print(df.to_string(index=False))
"""))

# ─── CELL 15: Defense comparison bar chart ─────────────────────────────────────
cells.append(md("## Cell 15 — Defense Comparison Bar Chart"))
cells.append(code("""\
far_plot  = CFG["far_values"][1]   # FAR 1e-4
far_col   = f"TAR@FAR{far_plot:.0e}"
attacks   = df["attack"].unique()
defenses  = ["none", "gaussian", "LSD"]
colors    = ["#e74c3c", "#3498db", "#2ecc71"]
labels    = ["No defense", "Gaussian blur (paper)", "LSD (ours)"]

fig, axes = plt.subplots(1, len(attacks), figsize=(4.5*len(attacks), 4.5), sharey=True)
if len(attacks) == 1: axes = [axes]

for ax, atk in zip(axes, attacks):
    sub  = df[(df["attack"]==atk) & (df["fiva"]=="orig")]
    vals = []
    for d in defenses:
        row = sub[sub["defense"]==d]
        vals.append(float(row[far_col].values[0]) if len(row) else 0.0)
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=.6)
    ax.set_title(atk, fontweight="bold", fontsize=11)
    ax.set_ylabel(f"Identity leakage @ {far_col}" if ax is axes[0] else "")
    ax.set_ylim(0, max(vals)*1.35 + 0.005)
    ax.tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle(f"Identity Leakage ({far_col}) — lower is better", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/defense_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── CELL 16: Sigma sweep vs LSD ───────────────────────────────────────────────
cells.append(md("## Cell 16 — Figure 5 Equivalent: Gaussian σ Sweep vs LSD"))
cells.append(code("""\
# Reproduce Figure 5 and add LSD reference line.
far_strict = CFG["far_values"][2]   # strictest: 1e-5
atk_fn     = ATTACKS["pgd_linf"]
sur_enc    = encoders["cosface"]

sigma_tar = {}
for sig in tqdm(CFG["gauss_sigmas"], desc="Sigma sweep"):
    if sig == 0.0:
        dfn = None
    else:
        s   = sig
        dfn = lambda x, _s=s: kornia.filters.gaussian_blur2d(x, (int(_s*6)|1,int(_s*6)|1), (_s,_s))
    tar = measure_leakage(atk_fn, sur_enc, encoders["arcface"],
                          fiva, ff_loader, far_thresholds, defense=dfn, max_batches=8)
    sigma_tar[sig] = tar[far_strict]

tar_lsd_val = measure_leakage(atk_fn, sur_enc, encoders["arcface"],
                               fiva, ff_loader, far_thresholds,
                               defense=lsd_fn, max_batches=8)[far_strict]

plt.figure(figsize=(7, 4))
plt.plot(CFG["gauss_sigmas"], list(sigma_tar.values()), "o-",
         color="#3498db", lw=2, label="Gaussian blur (paper defense)")
plt.axhline(tar_lsd_val, color="#2ecc71", ls="--", lw=2,
            label=f"LSD (ours) = {tar_lsd_val:.4f}")
plt.xlabel("Gaussian σ", fontsize=12)
plt.ylabel(f"Identity leakage @ FAR {far_strict:.0e}", fontsize=12)
plt.title("Gaussian σ sweep vs Learned Spectral Defense", fontsize=13)
plt.legend(); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sigma_vs_lsd.png", dpi=150)
plt.show()

best_sig = min(sigma_tar, key=lambda s: sigma_tar[s])
print(f"Best Gaussian σ={best_sig}  leakage={sigma_tar[best_sig]:.4f}")
print(f"LSD             leakage={tar_lsd_val:.4f}")
"""))

# ─── CELL 17: Qualitative visualisation ────────────────────────────────────────
cells.append(md("## Cell 17 — Qualitative Visualisation"))
cells.append(code("""\
def to_np(t):
    \"\"\"[-1,1] tensor [3,H,W] -> uint8 HWC RGB.\"\"\"
    return ((t.clamp(-1,1).cpu().float()+1)/2*255).permute(1,2,0).numpy().astype("uint8")

imgs_v, _ = next(iter(ff_loader))
imgs_v = imgs_v[:4].to(device)

with torch.no_grad():
    xadv_v  = ATTACKS["pgd_linf"](imgs_v, encoders["cosface"])
    xg_v    = gauss_15(xadv_v)
    xl_v    = lsd(xadv_v)
    deid_c,_,_  = fiva(imgs_v)
    deid_a,_,_  = fiva(xadv_v)
    deid_g,_,_  = fiva(xg_v)
    deid_l,_,_  = fiva(xl_v)

cols   = [imgs_v, xadv_v, xg_v, xl_v, deid_c, deid_a, deid_g, deid_l]
titles = ["Original","+ PGD-L∞","+ Gaussian","+ LSD (ours)",
          "FIVA-clean","FIVA-attack","FIVA+Gauss","FIVA+LSD"]

fig, axes = plt.subplots(4, 8, figsize=(20,10))
for r in range(4):
    for c, (col, ttl) in enumerate(zip(cols, titles)):
        axes[r,c].imshow(to_np(col[r]))
        axes[r,c].axis("off")
        if r==0: axes[r,c].set_title(ttl, fontsize=9, fontweight="bold")
plt.suptitle("Row=sample   Col=processing stage", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/qualitative.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR}/qualitative.png")
"""))

# ── Write notebook ───────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}
with open("adv_deid_gpu.ipynb","w") as f:
    json.dump(nb, f, indent=1)
print(f"Done — {len(cells)} cells written.")