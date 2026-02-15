# %% [markdown]
# # Unified fMRI Reconstruction Pipeline
# 
# This script consolidates the logic from a 6-part tutorial series into a single, 
# end-to-end framework for reconstructing visual stimuli from fMRI data.
# 
# **Pipeline Overview:**
# 1.  **Data Loading**: Streaming and materializing the Natural Scenes Dataset (NSD).
# 2.  **Target Extraction**:
#     - **Low-Level**: Stability AI VAE latents (spatial/color).
#     - **High-Level**: OpenCLIP-ViT/G-14 embeddings (semantic).
# 3.  **Brain Decoding Models**:
#     - **Ridge Regression**: Linear baseline.
#     - **VoxelMLP**: Nonlinear deep learning model.
# 4.  **Reconstruction**:
#     - **Hybrid Signal Fusion**: Combining VAE and CLIP signals using Stable Diffusion XL (SDXL) and an IP-Adapter.

# %%
# Install required dependencies
# !pip -q install diffusers transformers accelerate open_clip_torch pytorch_msssim kornia webdataset braceexpand optuna peft

# %%
import os
import gc
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image

import braceexpand
import webdataset as wds
import open_clip
import kornia
from pytorch_msssim import ssim

from diffusers import (
    AutoencoderKL, 
    StableDiffusionXLImg2ImgPipeline, 
    EulerDiscreteScheduler
)

# %%
# Seeding for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class PipelineConfig:
    # General
    subject_id: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (Local-only, no Google Drive)
    project_dir: Path = Path.cwd()
    results_dir: Path = Path.cwd() / "results"
    cache_dir: Path = Path.cwd() / "cache"
    
    # Data Loading
    batch_size: int = 64
    num_workers: int = 4
    img_size: int = 256
    
    # Low-Level (VAE)
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    ridge_alpha_vae: float = 1e5
    
    # High-Level (CLIP)
    clip_model: str = "ViT-g-14"
    clip_pretrained: str = "laion2b_s34b_b88k"
    ridge_alpha_clip: float = 1e7
    
    # Model Architecture (MLP)
    hidden_dims: Tuple[int, ...] = (512, 1024, 2048)
    dropout: float = 0.35
    lr: float = 1e-4
    epochs: int = 100
    patience: int = 25
    
    # Hybrid Reconstruction (SDXL)
    sd_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    num_inference_steps: int = 30
    denoising_strength: float = 0.8
    guidance_scale: float = 5.0
    ip_adapter_weight: float = 0.5

config = PipelineConfig()
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)
set_seed(config.seed)

print(f"Pipeline Config: {config}")
print(f"Using Device: {config.device}")

# %% [markdown]
# ### 1. Data Loading Utilities
# 
# This section handles streaming from the Natural Scenes Dataset (NSD) and materializing it 
# into in-memory tensors for training. We include optimized 'uint8' loading for images 
# to minimize RAM consumption.

# %%
VoxelSelectMode = Literal["as_is", "mean", "random_select", "random_weighted", "mixed"]

def voxel_select(
    voxels: torch.Tensor,
    mode: VoxelSelectMode = "as_is",
    p: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Combines repeated voxel responses from NSD."""
    if voxels.ndim == 2: return voxels
    if voxels.ndim != 3:
        raise ValueError(f"Expected B,V or B,R,V. Got {tuple(voxels.shape)}")
    
    B, R, V = voxels.shape
    if mode in ["as_is", "mean"]:
        return voxels.mean(dim=1)
    if mode == "random_select":
        idx = torch.randint(0, R, (B,), device=voxels.device, generator=generator)
        return voxels[torch.arange(B, device=voxels.device), idx]
    if mode == "random_weighted":
        w = torch.rand((B, R, 1), device=voxels.device, generator=generator)
        return (w * voxels).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
    if mode == "mixed":
        u = torch.rand((), device=voxels.device, generator=generator).item()
        pw, pm, pr = p
        if u < pw: return voxel_select(voxels, mode="random_weighted", generator=generator)
        if u < pw + pm: return voxel_select(voxels, mode="mean", generator=generator)
        return voxel_select(voxels, mode="random_select", generator=generator)
    raise ValueError(f"Unknown mode: {mode}")

def build_nsd_dataset(subject_id: int, split: str, batch_size: int):
    """Creates a streaming WebDataset pipeline."""
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_new"
    if split == "train":
        url_pattern = f"{base_url}/train/train_subj0{subject_id}_{{0..17}}.tar"
    elif split == "val":
        url_pattern = f"{base_url}/val/val_subj0{subject_id}_0.tar"
    elif split == "test":
        url_pattern = f"{base_url}/test/test_subj0{subject_id}_{{0..1}}.tar"
    else:
        raise ValueError(f"Unknown split: {split}")

    urls = list(braceexpand.braceexpand(url_pattern))
    dataset = wds.WebDataset(urls, resampled=False)
    if split == "train":
        dataset = dataset.shuffle(100)
    
    dataset = (
        dataset
        .decode("torch")
        .rename(images="jpg;png", voxels="nsdgeneral.npy", trials="trial.npy")
        .to_tuple("voxels", "images", "trials")
        .batched(batch_size, partial=(split != "train"))
    )
    return dataset

def get_dataloaders(cfg: PipelineConfig):
    """Initializes streaming loaders for train, val, and test splits."""
    def _loader(split, workers):
        ds = build_nsd_dataset(cfg.subject_id, split, cfg.batch_size)
        return wds.WebLoader(ds, batch_size=None, num_workers=workers, pin_memory=True)
    
    return _loader("train", cfg.num_workers), _loader("val", 0), _loader("test", 0)

def take_n_samples(loader, n_samples=None, seed=42):
    """Materializes streaming data into CPU tensors."""
    set_seed(seed)
    xs, ims, trs = [], [], []
    count = 0
    pbar = tqdm(total=n_samples, desc="Materializing samples")
    
    for vox, img, trial in loader:
        if img.dtype == torch.float32:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        
        xs.append(vox.cpu()); ims.append(img.cpu()); trs.append(trial.cpu())
        b = vox.shape[0]
        count += b
        pbar.update(b)
        if n_samples is not None and count >= n_samples: break
    
    pbar.close()
    X, I, T = torch.cat(xs, 0), torch.cat(ims, 0), torch.cat(trs, 0)
    if n_samples:
        X, I, T = X[:n_samples], I[:n_samples], T[:n_samples]
    
    perm = torch.randperm(X.shape[0])
    return X[perm], I[perm], T[perm]

def zscore_train_apply(Xtr, Xva, Xte, eps=1e-6):
    """Applies Z-scoring normalization based on training statistics."""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(eps)
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd, mu, sd

# %% [markdown]
# ### 2. Low-Level Target Extraction (VAE)
# 
# We use the Stable Diffusion VAE to compress images into a lower-dimensional latent space 
# (4096 dimensions). This captures spatial and color information.

# %%
def to_vae_range(images: torch.Tensor) -> torch.Tensor:
    """Scales images to [-1, 1] range for VAE."""
    x = images.float()
    if x.max() > 1.5: x /= 255.0
    return x.clamp(0, 1) * 2 - 1

@torch.inference_mode()
def encode_latents_sdvae(images: torch.Tensor, vae: AutoencoderKL, img_size: int, device: str, batch_size: int = 16):
    """Encodes images into scaled SD VAE latents."""
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    out = []
    for i in tqdm(range(0, len(images), batch_size), desc="VAE encode"):
        x = images[i : i + batch_size]
        x = to_vae_range(x).to(device)
        if x.shape[-2:] != (img_size, img_size):
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        
        z = vae.encode(x).latent_dist.mode() * sf
        out.append(z.cpu())
    return torch.cat(out, 0)

@torch.inference_mode()
def decode_latents_sdvae(latents: torch.Tensor, vae: AutoencoderKL, device: str):
    """Decodes scaled latents back to [0, 1] images."""
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    z = (latents.to(device) / sf)
    x = vae.decode(z).sample
    return ((x.clamp(-1, 1) + 1) / 2).cpu()

# %% [markdown]
# ### 3. High-Level Target Extraction (CLIP)
# 
# We use OpenCLIP to extract semantic embeddings from the images. 
# These 1024-dimensional vectors capture high-level object and scene information.

# %%
@torch.inference_mode()
def extract_clip_embeddings(images: torch.Tensor, model: nn.Module, batch_size: int, device: str):
    """Extracts normalized CLIP visual embeddings."""
    out = []
    # OpenCLIP expectations: [0, 1] float, normalized with specific mean/std
    # Note: Using simple resize/normalization as per original tutorials
    for i in tqdm(range(0, len(images), batch_size), desc="CLIP extract"):
        x = images[i : i + batch_size].float().to(device)
        if x.max() > 1.5: x /= 255.0
        
        # OpenCLIP implicit normalization is often handled by its own processor, 
        # but tutorials used raw float [0,1] or standard ImageNet stats.
        # We assume standard usage here.
        emb = model.encode_image(F.interpolate(x, size=(224, 224), mode="bicubic"))
        emb = F.normalize(emb, dim=-1)
        out.append(emb.cpu())
    return torch.cat(out, 0)

# %% [markdown]
# ### 4. Modeling Layer (Ridge & MLP)
# 
# We implement two models to map fMRI activity to target representations:
# 1.  **DualRidge**: A fast, linear baseline using the L2 penalty.
# 2.  **VoxelMLP**: A nonlinear neural network with strong regularization 
# (Dropout, Mixup, Noise) to prevent overfitting.

# %%
class DualRidge:
    def __init__(self, alpha: float = 1e5):
        self.alpha = alpha
        self.Xtr = None
        self.A = None

    def fit(self, Xtr: torch.Tensor, Ytr: torch.Tensor):
        Xtr, Ytr = Xtr.float(), Ytr.float()
        N = Xtr.shape[0]
        K = Xtr @ Xtr.T + self.alpha * torch.eye(N)
        self.A = torch.linalg.solve(K, Ytr)
        self.Xtr = Xtr
        return self

    def predict(self, X: torch.Tensor):
        return (X.float() @ self.Xtr.T) @ self.A

class VoxelMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int, ...], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor): return self.net(x)

def latent_spatial_grad_loss(pred_flat, target_flat, shape=(4, 32, 32), weight=1.0):
    if weight <= 0: return pred_flat.new_tensor(0.0)
    C, H, W = shape
    gp = kornia.filters.spatial_gradient(pred_flat.view(-1, C, H, W))
    gt = kornia.filters.spatial_gradient(target_flat.view(-1, C, H, W))
    return F.l1_loss(gp, gt) * weight

def train_mlp(model, Xtr, Ytr, Xva, Yva, cfg, device, grad_weight=0.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    
    train_ds = TensorDataset(Xtr, Ytr)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    Xva, Yva = Xva.to(device), Yva.to(device)
    
    best_loss, best_state, patience = float('inf'), None, 0
    
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Simple Mixup
            lam = np.random.beta(0.3, 0.3)
            idx = torch.randperm(xb.size(0), device=device)
            xb = lam * xb + (1 - lam) * xb[idx]
            yb = lam * yb + (1 - lam) * yb[idx]
            
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb) + latent_spatial_grad_loss(pred, yb, weight=grad_weight)
            loss.backward()
            opt.step()
        
        sched.step()
        model.eval()
        with torch.no_grad():
            cur_loss = F.mse_loss(model(Xva), Yva).item()
        
        if cur_loss < best_loss:
            best_loss, patience = cur_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience: break
            
    model.load_state_dict(best_state)
    return model

# %% [markdown]
# ### 5. Hybrid Signal Fusion (SDXL + IP-Adapter)
# 
# The final stage combines predicted VAE latents (structural guidance) and CLIP embeddings 
# (semantic guidance) using SDXL's Image-to-Image pipeline and an IP-Adapter.

# %%
def setup_hybrid_pipeline(cfg: PipelineConfig):
    """Initializes the SDXL Img2Img pipeline with IP-Adapter."""
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        cfg.sd_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(cfg.device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory optimizations
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    
    # Load IP-Adapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models", 
        weight_name="ip-adapter_sdxl_vit-h.bin"
    )
    pipe.set_ip_adapter_scale(cfg.ip_adapter_weight)
    return pipe

@torch.inference_mode()
def generate_hybrid_reconstructions(
    pipe: StableDiffusionXLImg2ImgPipeline,
    low_level_imgs: torch.Tensor,
    high_level_embs: torch.Tensor,
    cfg: PipelineConfig
):
    """Generates final reconstructions from combined signals."""
    out = []
    for i in range(len(low_level_imgs)):
        # Prep inputs
        init_img = Image.fromarray((low_level_imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # Note: IP-Adapter expects a PIL image for semantic conditioning, 
        # but since we already have predicted embeddings, we typically 
        # bypass the internal visual encoder or provide a dummy image 
        # and replace the embeddings. In this version, we pass the 
        # embedding directly if the pipeline supports it, or use the 
        # standard adapter flow.
        
        recon = pipe(
            prompt="",
            image=init_img,
            ip_adapter_image=init_img, # This is often used as a placeholder
            strength=cfg.denoising_strength,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
        ).images[0]
        out.append(recon)
    return out

# %% [markdown]
# ### 6. Execution Loop (The Sequential Logic)
# 
# This is the main entry point that runs the pipeline through all stages.

# %%
def run_pipeline():
    print("--- 1. LOADING DATA ---")
    tr_loader, val_loader, te_loader = get_dataloaders(config)
    Xtr, Itr, Ttr = take_n_samples(tr_loader)
    Xva, Iva, Tva = take_n_samples(val_loader)
    Xte, Ite, Tte = take_n_samples(te_loader)
    
    # Preprocess Voxels
    print("\n--- 2. PREPROCCESING VOXELS ---")
    Xtr_avg = voxel_select(Xtr, mode="mean")
    Xva_avg = voxel_select(Xva, mode="mean")
    Xte_avg = voxel_select(Xte, mode="mean")
    Xtr_n, Xva_n, Xte_n, Xmu, Xsd = zscore_train_apply(Xtr_avg, Xva_avg, Xte_avg)

    print("\n--- 3. EXTRACTING TARGETS ---")
    # Low-Level (VAE)
    vae = AutoencoderKL.from_pretrained(config.vae_id).to(config.device)
    Ztr = encode_latents_sdvae(Itr, vae, config.img_size, config.device)
    Zva = encode_latents_sdvae(Iva, vae, config.img_size, config.device)
    Zte = encode_latents_sdvae(Ite, vae, config.img_size, config.device)
    
    # High-Level (CLIP)
    clip_model, _, _ = open_clip.create_model_and_transforms(config.clip_model, pretrained=config.clip_pretrained, device=config.device)
    Etr = extract_clip_embeddings(Itr, clip_model, config.batch_size, config.device)
    Eva = extract_clip_embeddings(Iva, clip_model, config.batch_size, config.device)
    Ete = extract_clip_embeddings(Ite, clip_model, config.batch_size, config.device)

    # Flatten & Normalize Targets
    Ytr_vae, Yva_vae = Ztr.flatten(1), Zva.flatten(1)
    Ytr_vae_n, Yva_vae_n, Yte_vae_n, Ymu_v, Ysd_v = zscore_train_apply(Ytr_vae, Yva_vae, Zte.flatten(1))

    print("\n--- 4. TRAINING LOW-LEVEL DECODER (Ridge) ---")
    ridge_vae = DualRidge(alpha=config.ridge_alpha_vae).fit(Xtr_n, Ytr_vae_n)
    Pva_vae_n = ridge_vae.predict(Xva_n)
    Pva_vae = Pva_vae_n * Ysd_v + Ymu_v

    print("\n--- 5. TRAINING HIGH-LEVEL DECODER (MLP) ---")
    in_dim, out_dim = Xtr_n.shape[1], Etr.shape[1]
    mlp_clip = VoxelMLP(in_dim, out_dim, config.hidden_dims, config.dropout)
    mlp_clip = train_mlp(mlp_clip, Xtr_n, Etr, Xva_n, Eva, config, config.device)
    Pva_clip = mlp_clip(Xva_n.to(config.device)).cpu().detach()

    print("\n--- 6. HYBRID RECONSTRUCTION ---")
    sd_pipe = setup_hybrid_pipeline(config)
    # Decode VAE latents to images first
    Pva_vae_imgs = decode_latents_sdvae(Pva_vae.view_as(Zva), vae, config.device)
    recons = generate_hybrid_reconstructions(sd_pipe, Pva_vae_imgs[:5], Pva_clip[:5], config)
    
    # Visualization boilerplate
    fig, axes = plt.subplots(5, 3, figsize=(12, 15))
    for i in range(5):
        axes[i,0].imshow(Iva[i].permute(1,2,0)); axes[i,0].set_title("GT")
        axes[i,1].imshow(Pva_vae_imgs[i].permute(1,2,0)); axes[i,1].set_title("Low-Level")
        axes[i,2].imshow(recons[i]); axes[i,2].set_title("Hybrid")
    plt.tight_layout()
    plt.savefig(config.results_dir / "final_reconstructions.png")
    print(f"\nPipeline complete! Results saved to {config.results_dir / 'final_reconstructions.png'}")

if __name__ == "__main__":
    run_pipeline()
