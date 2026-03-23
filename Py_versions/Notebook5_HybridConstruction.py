#!/usr/bin/env python
# coding: utf-8

# # Notebook 5: Hybrid Reconstruction
# 
# In the previous notebooks, we built two complementary decoding pipelines. Notebook 3 produced blurry reconstructions that preserve spatial structure and color layout. Notebook 4 predicted semantic embeddings that capture what objects and scenes are present. Now we bring these together.
# 
# ## The Core Idea
# 
# Neither low-level nor high-level decoding alone is sufficient for high-quality reconstruction. The VAE reconstructions from Notebook 3 get the rough layout right but lack fine details and textures. The CLIP embeddings from Notebook 4 know what should be in the image but cannot produce pixels on their own. The solution is to combine both signals using a powerful generative model.
# 
# This is exactly how state-of-the-art methods like MindEye and Brain-Diffuser work. They treat the low-level reconstruction as a structural prior and the semantic embedding as a content guide. A diffusion model then synthesizes the final image by adding realistic details that are consistent with both signals.
# 
# ## What Happens in This Notebook
# 
# We will use Stable Diffusion XL in image-to-image mode. The pipeline takes the blurry reconstruction from Notebook 3 as input and progressively refines it through the denoising process. Simultaneously, an IP-Adapter injects the predicted CLIP embedding to steer generation toward semantically appropriate content.
# 
# The mathematical intuition is straightforward. Standard diffusion models sample from $p(x \mid \text{prompt})$. In our case, we sample from:
# 
# $$p(x \mid I_{\text{low-level}}, C_{\text{CLIP}})$$
# 
# where $I_{\text{low-level}}$ is the blurry reconstruction and $C_{\text{CLIP}}$ is the predicted semantic embedding. The low-level image provides the starting point in latent space, while the CLIP embedding guides the denoising trajectory.
# 
# ## A Note on Hardware
# 
# While cutting-edge research often uses clusters of A100 or H100 GPUs, we can still achieve meaningful results on more modest hardware like a T4 GPU with 16GB VRAM. The key is using memory-efficient techniques like fp16 precision and VAE tiling that let us run the full SDXL model without running out of memory.
# 
# ## What You Will Learn
# 
# By the end of this notebook, you will understand how modern reconstruction pipelines fuse low-level and high-level brain signals. You will see how the IP-Adapter allows us to condition image generation directly on brain-derived embeddings without needing text prompts. You will also learn practical techniques for running large diffusion models on limited hardware. Most importantly, you will see how the combination of both signals produces reconstructions that neither could achieve alone.

# In[1]:


get_ipython().system('pip -q install --upgrade diffusers transformers accelerate')


# In[2]:


import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
from diffusers import StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler, AutoencoderKL
import random
import subprocess
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import (
    inception_v3,
    Inception_V3_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim


try:
    import clip
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git"]
    )
    import clip




# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# In[3]:


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# In[4]:


if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        ROOT_PATH = "/content/drive/MyDrive/NSD_Results"
        RECON_PATH = "/content/drive/MyDrive/NSD_Reconstructions/temp_decode"
    except ImportError:
        ROOT_PATH = "./results"
        RECON_PATH = "./NSD_Reconstructions/temp_decode"
else:
    ROOT_PATH = "./results"
    RECON_PATH = "./NSD_Reconstructions/temp_decode"

print(f"Loading results from: {ROOT_PATH}")


# ## 1. Loading Our Previous Results
# 
# Before we can fuse the two signals, we need to load the outputs from both previous notebooks.
# 
# From Notebook 3, we have the low-level reconstructions. These are 256x256 RGB images decoded from the VAE latents we predicted from brain activity. They capture the rough spatial structure, dominant colors, and approximate object positions, but lack fine details and textures.
# 
# From Notebook 4, we have the predicted OpenCLIP embeddings. Each embedding is a 1024-dimensional vector that encodes the semantic content of what the subject was viewing. These embeddings live in the same space as CLIP text embeddings, which is what makes them useful for guiding image generation.
# 
# The code below loads both sets of predictions and ensures they are properly aligned so that each low-level reconstruction is paired with its corresponding semantic embedding.

# In[5]:


high_level_path = os.path.join(ROOT_PATH, "high_level_test_results.pt")
recon_images_path = RECON_PATH

if not os.path.exists(high_level_path):
    raise FileNotFoundError(f"Missing {high_level_path}. Please run Notebook 4.")

nb4_results = torch.load(high_level_path, map_location="cpu")

# Handle dictionary
if isinstance(nb4_results, dict):
    pred_key = None
    for candidate_key in ("mlp", "pred_clip_mlp"):
        if candidate_key in nb4_results:
            pred_key = candidate_key
            break
    if pred_key is None:
        available = sorted(nb4_results.keys())
        raise KeyError(
            "Notebook 4 output missing predicted CLIP embeddings. "
            "Expected one of {'mlp', 'pred_clip_mlp'}, got keys: "
            f"{available}"
        )
    C_pred_hl = torch.as_tensor(nb4_results[pred_key])

    gt_key = None
    for candidate_key in ("gt_images", "test_images"):
        if candidate_key in nb4_results:
            gt_key = candidate_key
            break
    if gt_key is None:
        available = sorted(nb4_results.keys())
        raise KeyError(
            "Notebook 4 output missing ground-truth test images. "
            "Expected one of {'gt_images', 'test_images'}, got keys: "
            f"{available}"
        )
    gt_images_tensor = torch.as_tensor(nb4_results[gt_key])
else:
    C_pred_hl = torch.as_tensor(nb4_results)
    raise TypeError(
        "Notebook 4 output should be a dict with explicit keys "
        "for predicted embeddings and test images."
    )

# Load Recons
if os.path.exists(recon_images_path):
    recon_files = [
        f for f in os.listdir(recon_images_path)
        if f.startswith("test_recon_batch") and f.endswith(".pt")
    ]
else:
    recon_files = []

# Fallback local
if not recon_files and os.path.exists("./results/test_recon_batch_0.pt"):
    recon_files = ["test_recon_batch_0.pt"]
    recon_images_path = "./results"

if not recon_files:
    raise FileNotFoundError(
        "No reconstruction batches found. Please run Notebook 3 first."
    )

def _batch_index(filename):
    match = re.search(r"test_recon_batch_(\d+)\.pt$", filename)
    return int(match.group(1)) if match else -1

recon_files = sorted(recon_files, key=_batch_index)
if any(_batch_index(f) < 0 for f in recon_files):
    raise ValueError(
        "Unexpected reconstruction file names. "
        "Expected format: test_recon_batch_<idx>.pt"
    )

batch_indices = [_batch_index(f) for f in recon_files]
if batch_indices[0] != 0:
    raise ValueError(
        "Reconstruction batches should start at index 0. "
        f"Found first batch index {batch_indices[0]}."
    )

# Validate contiguity in sample-index space using actual loaded batch sizes
recon_imgs_list = []
expected_start = 0
for f_name in recon_files:
    file_start = _batch_index(f_name)
    if file_start != expected_start:
        raise ValueError(
            "Reconstruction batches are not contiguous in sample index space. "
            f"File {f_name} starts at {file_start}, expected {expected_start}."
        )

    tensor_batch = torch.as_tensor(
        torch.load(os.path.join(recon_images_path, f_name), map_location="cpu")
    )
    if tensor_batch.ndim != 4:
        raise ValueError(
            f"{f_name} must be 4D [batch, channels, height, width], "
            f"got shape {tuple(tensor_batch.shape)}"
        )

    recon_imgs_list.append(tensor_batch)
    expected_start += tensor_batch.shape[0]

I_pred_ll = torch.cat(recon_imgs_list, dim=0)

# Strict alignment checks just for safe measure
n_ll = len(I_pred_ll)
n_hl = len(C_pred_hl)
n_gt = len(gt_images_tensor)
if n_ll != n_hl or n_ll != n_gt:
    raise ValueError(
        "Notebook handoff mismatch detected. "
        f"low-level={n_ll}, high-level={n_hl}, gt={n_gt}. "
        "Please re-run Notebook 3 and Notebook 4 end-to-end with matching test splits."
    )

print(f"Loaded and verified {n_ll} aligned samples.")


# ## 2. Loading the SDXL Pipeline
# 
# Stable Diffusion XL is a powerful text-to-image diffusion model with 3.5 billion parameters. We use it in image-to-image mode, which means instead of starting from pure noise, we start from an encoded version of our low-level reconstruction and denoise from there.
# 
# The pipeline has several components working together. The VAE encoder compresses our input image into a latent representation. The U-Net performs iterative denoising in this latent space. The VAE decoder then converts the final latent back into a full-resolution image. We use a specialized fp16-compatible VAE from madebyollin that avoids numerical instability issues when running in half precision.
# 
# The IP-Adapter is a lightweight module that allows us to condition generation on image embeddings rather than text. It was trained to accept CLIP image embeddings and inject them into the cross-attention layers of the U-Net, effectively telling the model what semantic content should appear. In our case, we feed it the predicted brain embeddings from Notebook 4, which were trained to match OpenCLIP ViT-H embeddings.
# 
# We apply several memory optimizations to fit SDXL within our GPU memory budget. VAE tiling breaks the image into smaller tiles during encoding and decoding, dramatically reducing peak memory usage. We also use fp16 precision throughout, cutting memory requirements roughly in half compared to fp32.

# In[6]:


print("Loading SDXL Components...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "pipe" in globals():
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 1) VAE
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# 2) Base pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    low_cpu_mem_usage=True
)

# 3) Load IP-Adapter weights
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl_vit-h.safetensors"
)

# 4) Scheduler + device + memory-safe VAE
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_vae_tiling()


# Sanity check: cross-attn processors should be IPAdapter-aware
bad = []
for name, proc in pipe.unet.attn_processors.items():
    if "attn2" in name and "IPAdapter" not in proc.__class__.__name__:
        bad.append((name, proc.__class__.__name__))

if bad:
    raise RuntimeError(
        "IP-Adapter processors were overwritten. Restart runtime and rerun this cell only. "
        f"Examples: {bad[:3]}"
    )

print("SDXL Img2Img + IP-Adapter ready.")


# ## 3. The Generation Function
# 
# The generation function ties everything together. It takes a low-level reconstruction and a semantic embedding, then produces a high-quality output image.
# 
# The first step is preprocessing. Our low-level reconstructions are 256x256, but SDXL works best at its native 1024x1024 resolution. We upscale using Lanczos resampling, which is a high-quality interpolation method that preserves edges better than bilinear or bicubic alternatives.
# 
# Next we prepare the semantic conditioning. The IP-Adapter expects embeddings in a specific format, so we reshape the predicted CLIP vector and pair it with a zero embedding for classifier-free guidance. Unlike our training loss in Notebook 4, we do not renormalize the embedding to unit length here. Instead, we preserve its natural scale so the conditioning more closely matches the distribution the IP-Adapter saw during training.
# 
# 

# #### Generation Functions
# Here we define the two generation conditions used in the comparison. The semantic-only baseline tests what the high-level embedding can do by itself, while the hybrid function combines that semantic signal with the low-level reconstruction from Notebook 3.

# In[23]:


def preprocess_image(tensor_img):
    t = tensor_img.clone().detach().cpu()
    if t.min() < 0:
        t = (t + 1) / 2
    t = (t * 255).byte() if t.max() <= 1.0 else t.byte()
    t = t.permute(1, 2, 0).numpy()
    return Image.fromarray(t)


def prepare_ip_adapter_embeds(embedding_tensor):
    """Prepare CLIP embedding for IP-Adapter.

    IP-Adapter's internal cross-attention layers were trained on real CLIP
    embeddings with their natural magnitude (~20 for ViT-H/14). We preserve
    that magnitude rather than collapsing to unit length. The embedding is
    reshaped to [2, 1, D] with zeros as the negative (unconditional) embed
    for classifier-free guidance.
    """
    emb = embedding_tensor.view(1, 1, -1).to(dtype=torch.float16, device=device)
    neg_embeds = torch.zeros_like(emb)
    return torch.cat([neg_embeds, emb], dim=0)


@torch.no_grad()
def generate_sdxl_2pass(
    low_level_img_tensor,
    embedding_tensor,
    seed=42,
    pass1_steps=24,
    pass2_steps=28,
    pass1_strength=0.7,
    pass2_strength=0.35,
):
    pil_low = preprocess_image(low_level_img_tensor)
    pil_low = pil_low.resize((1024, 1024), Image.LANCZOS)

    combined_embeds = prepare_ip_adapter_embeds(embedding_tensor)

    # Pass 1: semantic correction while preserving low-level layout
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.set_ip_adapter_scale(1.1)
    pass1_result = pipe(
        prompt="realistic",
        negative_prompt="blurry, unrealistic, extra limbs",
        image=pil_low,
        ip_adapter_image_embeds=[combined_embeds],
        strength=pass1_strength,
        num_inference_steps=pass1_steps,
        guidance_scale=6.5,
        generator=generator,
    ).images[0]

    # Pass 2: refinement directly on pass1 output
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.set_ip_adapter_scale(0.3)
    final_result = pipe(
        prompt="high quality, highly detailed, sharp focus, realistic, hyperrealistic, sharp",
        negative_prompt="blurry, low quality, distorted, unrealistic, painting, illustration, fog, smoke,blur",
        image=pass1_result,
        ip_adapter_image_embeds=[combined_embeds],
        strength=pass2_strength,
        num_inference_steps=pass2_steps,
        guidance_scale=6,
        generator=generator,
    ).images[0]

    del pass1_result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_result, pil_low


@torch.no_grad()
def generate_semantic_only(embedding_tensor, seed=42, steps=30):
    blank = Image.new("RGB", (1024, 1024), (127, 127, 127))

    combined_embeds = prepare_ip_adapter_embeds(embedding_tensor)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.set_ip_adapter_scale(1.5)
    result = pipe(
        prompt="sharp, realistic, photographic",
        negative_prompt="blurry, low quality, distorted, unrealistic, painting, illustration",
        image=blank,
        ip_adapter_image_embeds=[combined_embeds],
        strength=1,
        num_inference_steps=steps,
        guidance_scale=8,
        generator=generator,
    ).images[0]
    return result


# ## 4.1. Visualizing the Results
# 
# Now we can see the complete pipeline in action. For each test sample, we display three images side by side.
# 
# The ground truth shows the original image that was presented to the subject during the fMRI scan. This is what we are trying to reconstruct from brain activity alone.
# 
# The low-level input shows the reconstruction from Notebook 3 after upscaling to 1024x1024. You can see that it captures the general colors and spatial layout but appears blurry and lacks fine details. Edges are soft, textures are missing, and small objects are often unrecognizable.
# 
# The final output shows what SDXL produces when guided by both the low-level structure and the semantic embedding. The diffusion model adds realistic textures, sharpens edges, and fills in plausible details that are consistent with the predicted semantics. Objects become more recognizable and the overall image quality improves dramatically.
# 
# Pay attention to how the combination works. The spatial layout from the low-level reconstruction constrains where things appear. The semantic embedding influences what those things look like. Neither signal alone determines the output; rather, they work together to produce something better than either could achieve independently.

# In[24]:


torch.manual_seed(42)
indices = torch.randperm(len(I_pred_ll))[:10].tolist()

fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5*len(indices)))

for i, idx in enumerate(indices):
    ll_img = I_pred_ll[idx]
    brain_emb = C_pred_hl[idx]

    # Generate with 2-pass method only
    result_2pass, input_low_1024 = generate_sdxl_2pass(ll_img, brain_emb, seed=28)

    if gt_images_tensor is not None:
        gt_pil = preprocess_image(gt_images_tensor[idx])
    else:
        gt_pil = Image.new('RGB', (256, 256))

    axes[i, 0].imshow(gt_pil)
    axes[i, 0].set_title("Ground Truth", fontsize=12)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(input_low_1024)
    axes[i, 1].set_title("Low-Level Input\n(Notebook 3)", fontsize=12)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(result_2pass)
    axes[i, 2].set_title("Final Reconstruction\n", fontsize=12)
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()


# ## 4.2. Quantitative Evaluation
# 
# The visual comparisons above give a sense of reconstruction quality, but we also want numbers. This cell computes four metrics across three conditions to quantify whether the hybrid combination is doing what we claim.
# 
# The three conditions are **low-level only** (the blurry VAE reconstruction from Notebook 3 with no semantic guidance), **semantic only** (SDXL conditioned on the predicted CLIP embedding alone with no structural input), and **hybrid** (the full pipeline combining both signals).
# 
# We measure each against the ground truth using four metrics that span two levels of similarity. **PixCorr** and **SSIM** measure pixel-level fidelity, capturing how closely the reconstruction matches the original in spatial structure and luminance. **InceptionV3** and **CLIP** measure semantic similarity using 2-way identification accuracy: given a reconstruction and two candidate images (one correct, one random distractor), how often does the model's feature space pick the correct match? Chance performance is 50%.
# 
# For the tutorial walkthrough, the evaluation runs on a small subset to keep Colab runtime manageable. For the paper, we evaluate on the full 1,000-image test set. All pixel-level metrics are computed at 425×425 resolution following the standard evaluation protocol in the reconstruction literature.
# 
# We expect a consistent trade-off pattern: low-level should dominate on pixel metrics, semantic-only should dominate on high-level metrics, and hybrid should find a productive middle ground between the two.
# 

# #### Image Helper
# 
# These small utilities prepare images for evaluation. The first converts a PIL image into a normalized PyTorch tensor, and the second defines the resolution used for our quantitative metrics.

# In[25]:


def pil_to_chw01(pil_img, size=256):
    pil_img = pil_img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


EVAL_RESOLUTION = 425


# #### Evaluation Metrics
# We evaluate reconstructions at two levels. Pixel-based metrics such as PixCorr and SSIM measure spatial fidelity, while feature-based identification metrics ask whether a pretrained vision model still recognizes the reconstruction as matching the correct image.

# In[26]:


@torch.no_grad()
def extract_features_batched(images, model, preprocess, feature_layer=None, batch_size=8, device=device):
    feats = []
    for start_idx in range(0, len(images), batch_size):
        batch_imgs = images[start_idx:start_idx + batch_size]
        batch = torch.stack([preprocess(img) for img in batch_imgs], dim=0).to(device)
        outputs = model(batch)
        outputs = outputs if feature_layer is None else outputs[feature_layer]
        feats.append(outputs.float().flatten(1).cpu())
        del batch, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0).numpy()


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, batch_size=8, device=device):
    preds = extract_features_batched(
        all_recons, model, preprocess, feature_layer=feature_layer, batch_size=batch_size, device=device
    )
    reals = extract_features_batched(
        all_images, model, preprocess, feature_layer=feature_layer, batch_size=batch_size, device=device
    )
    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, axis=0)
    return float(np.mean(success_cnt) / (len(all_images) - 1))


def compute_pixcorr(all_recons, all_images):
    resize = transforms.Resize(EVAL_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR)
    gt_flat = resize(all_images).reshape(len(all_images), -1).cpu().numpy()
    recon_flat = resize(all_recons).reshape(len(all_recons), -1).cpu().numpy()
    corrs = [np.corrcoef(gt_flat[i], recon_flat[i])[0, 1] for i in range(len(gt_flat))]
    return float(np.nanmean(corrs))


def compute_ssim_mean(all_recons, all_images):
    resize = transforms.Resize(EVAL_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR)
    gt_resized = resize(all_images).permute(0, 2, 3, 1).cpu().numpy()
    recon_resized = resize(all_recons).permute(0, 2, 3, 1).cpu().numpy()
    scores = []
    for gt_img, recon_img in zip(gt_resized, recon_resized):
        gt_gray = rgb2gray(gt_img)
        recon_gray = rgb2gray(recon_img)
        scores.append(
            ssim(recon_gray, gt_gray, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, data_range=1.0, channel_axis=None)
        )
    return float(np.mean(scores))


# ####Run the Tutorial Evaluation
# This cell runs a small evaluation subset to keep the tutorial practical on Colab. In the paper we report full test-set numbers, but here the goal is to show the expected trade-off pattern between low-level, semantic-only, and hybrid reconstructions without a very long runtime.

# In[30]:


N_QUANT_EVAL = 40
#N_QUANT_EVAL = len(gt_images_tensor) #Uncomment this line to run the full evaluation, same with the paper.
SEM_ONLY_STEPS = 8
HYB_PASS1_STEPS = 12
HYB_PASS2_STEPS = 12
BASE_SEED = 42

eval_indices = list(range(N_QUANT_EVAL))
print(f"Quantitative evaluation: {N_QUANT_EVAL} test samples")

gt_imgs = []
low_imgs = []
sem_imgs = []
hyb_imgs = []

for i, idx in enumerate(eval_indices, 1):
    gt_pil = preprocess_image(gt_images_tensor[idx])
    low_pil = preprocess_image(I_pred_ll[idx])

    sem_only_pil = generate_semantic_only(
        C_pred_hl[idx], seed=BASE_SEED + idx, steps=SEM_ONLY_STEPS,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hybrid_pil, _ = generate_sdxl_2pass(
        I_pred_ll[idx], C_pred_hl[idx], seed=BASE_SEED + idx,
        pass1_steps=HYB_PASS1_STEPS, pass2_steps=HYB_PASS2_STEPS,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gt_imgs.append(pil_to_chw01(gt_pil, size=256))
    low_imgs.append(pil_to_chw01(low_pil, size=256))
    sem_imgs.append(pil_to_chw01(sem_only_pil, size=256))
    hyb_imgs.append(pil_to_chw01(hybrid_pil, size=256))

    del gt_pil, low_pil, sem_only_pil, hybrid_pil
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if i % 10 == 0 or i == N_QUANT_EVAL:
        print(f"  Generated {i}/{N_QUANT_EVAL}")

# Compute metrics
all_gt = torch.stack(gt_imgs).float()
method_images = {
    "Low-level only": torch.stack(low_imgs).float(),
    "Semantic only": torch.stack(sem_imgs).float(),
    "Hybrid": torch.stack(hyb_imgs).float(),
}
del gt_imgs, low_imgs, sem_imgs, hyb_imgs
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Low-level metrics
summary = {
    name: {
        "PixCorr": compute_pixcorr(recons, all_gt),
        "SSIM": compute_ssim_mean(recons, all_gt),
    }
    for name, recons in method_images.items()
}

# InceptionV3 pairwise identification
inception_weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(
    inception_v3(weights=inception_weights), return_nodes=["avgpool"],
).to(device)
inception_model.eval().requires_grad_(False)
inception_preprocess = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
for name, recons in method_images.items():
    summary[name]["InceptionV3"] = two_way_identification(
        recons, all_gt, inception_model, inception_preprocess,
        feature_layer="avgpool", batch_size=8, device=device,
    )
del inception_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# CLIP pairwise identification
clip_model, _ = clip.load("ViT-L/14", device=device)
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])
for name, recons in method_images.items():
    summary[name]["CLIP"] = two_way_identification(
        recons, all_gt, clip_model.encode_image, clip_preprocess,
        feature_layer=None, batch_size=8, device=device,
    )
del clip_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Results
metric_order = ["PixCorr", "SSIM", "InceptionV3", "CLIP"]
summary_df = pd.DataFrame({
    "Metric": metric_order,
    "Low-level only": [summary["Low-level only"][m] for m in metric_order],
    "Semantic only": [summary["Semantic only"][m] for m in metric_order],
    "Hybrid": [summary["Hybrid"][m] for m in metric_order],
})

print(f"\nQuantitative summary ({N_QUANT_EVAL} samples, {EVAL_RESOLUTION}x{EVAL_RESOLUTION}):")
print(summary_df.to_string(index=False))


# The table evaluates all three conditions across four metrics that capture different levels of visual similarity. PixCorr and SSIM measure pixel-level fidelity, reflecting how closely the reconstruction matches the original in raw spatial structure and luminance. InceptionV3 and CLIP operate at a higher level, measuring whether the reconstruction depicts the same kind of scene and objects as the original. These two neural network metrics are reported as 2-way identification accuracy. Given a reconstruction and two candidate images, one correct and one random distractor, how often does the model's feature space pick the correct match? Chance performance is 50%.
# 
# The pattern across conditions is consistent. Low-level-only reconstructions are strong on pixel-level metrics because they directly preserve spatial structure from the decoded latents, but they struggle on high-level metrics because a blurry color-matched blob often lacks recognizable object identity. Semantic-only generation flips this entirely. High-level scores are strong, but pixel-level fidelity is weak because the generative model has no spatial anchor and invents layouts freely. The hybrid pipeline finds a middle ground, retaining much of the spatial structure while matching or exceeding the semantic-only condition on high-level metrics.
# 
# A caveat is that the CLIP embedding is the dominant signal in our hybrid reconstructions for this tutorial. The low-level path is still useful, but as Notebook 3 showed, VAE reconstructions from brain activity reliably recover coarse layout features like the separation of ground and sky while they are not as reliable in the other spatial details. Since the hybrid pipeline uses these low-level reconstructions as its structural starting point, the fidelity ceiling they impose carries forward into the final output. We encoruage you to see how the low level priors affect the final hybrid reconstructions by comparing the low-level and hybrid reconstructions below.
# 
# 

# In[34]:


# Visualize a grid of reconstructions
N_SHOW = N_QUANT_EVAL//3
fig, axes = plt.subplots(N_SHOW, 4, figsize=(16, 4 * N_SHOW))

col_labels = ["Ground Truth", "Low-level only", "Semantic only", "Hybrid"]

for i in range(N_SHOW):
    axes[i, 0].imshow(all_gt[i].permute(1, 2, 0).numpy())
    axes[i, 1].imshow(method_images["Low-level only"][i].permute(1, 2, 0).numpy())
    axes[i, 2].imshow(method_images["Semantic only"][i].permute(1, 2, 0).numpy())
    axes[i, 3].imshow(method_images["Hybrid"][i].permute(1, 2, 0).numpy())

    for j in range(4):
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(col_labels[j], fontsize=12)

plt.tight_layout()
plt.show()


# The reconstructions illustrate the complementary nature of the two signals. Low-level outputs preserve coarse layout and color but lack semantic content. Semantic-only outputs often identify the correct category but invent the spatial context entirely. Hybrid outputs combine both, placing recognizable objects in approximately correct positions with approximately correct colors, though fine details diverge from the ground truth. The combination does not benefit every sample equally, and this variability is more visible in our pipeline than in state-of-the-art systems because we lack several components that published pipelines use to bridge the gap between noisy brain-predicted embeddings and the conditioning interface of the generative model. These include dedicated diffusion priors, dual CLIP conditioning through both text and vision branches, and custom-trained generation models. These components are omitted because they could not be reliably implemented within the hardware constraints of this tutorial without substantially increasing complexity and runtime.
# 
# It is also worth noting that modern reconstruction pipelines often generate multiple candidate images per stimulus and select the best one using the model's own predicted embedding as a scoring criterion. MindEye, for example, generates 16 candidates per test image and selects the one whose CLIP embedding is most similar to the brain-predicted embedding. This second-order selection can substantially boost quantitative metrics because the stochastic nature of diffusion sampling means some candidates will align better with the target by chance. Our pipeline generates a single image per stimulus with a fixed seed, which provides a more conservative estimate of reconstruction quality but also means our reported metrics do not benefit from this selection effect.
# 
# 

# ## A Note on Hyperparameters
# 
# The generation stage is far more sensitive to its hyperparameters than the decoding stages. Small changes to diffusion strength, IP-Adapter scale, guidance scale, or the number of denoising steps can noticeably shift the balance between pixel-level fidelity and semantic accuracy, and the relationship between these parameters is not always intuitive. The values used in this notebook were not studied extensively. They were chosen to produce a reasonable balance across the four evaluation metrics, but different configurations can produce noticeably better or worse results depending on what you optimize for.
# 
# To help orient your experimentation, here is a rough guide to what each parameter controls. **Diffusion strength** determines how much the model is allowed to change the input image. A strength of 0.0 would return the input unchanged, while 1.0 would ignore it entirely and generate from scratch. Higher values produce sharper and more detailed outputs but risk overriding the spatial structure recovered from the brain signal. **IP-Adapter scale** controls how strongly the CLIP embedding influences generation. Higher values push the output more aggressively toward the predicted semantic content, which improves object identity but can introduce visual artifacts or override layout if set too high. **Guidance scale** amplifies the difference between the conditioned and unconditioned predictions during denoising. Higher values produce outputs that follow the conditioning signal more closely but can look oversaturated or unnatural at extremes. **Number of denoising steps** controls how many iterations the diffusion process runs. More steps generally produce finer detail at the cost of longer runtime, with diminishing returns beyond a certain point.
# 
# We encourage you to experiment with these settings yourself. In our run we tried to focus more on the semantic structure for the hybrid architecture. Adjusting hyperparameters is one of the easiest ways to build intuition for how the generative stage works and to see firsthand how much the final output depends on decisions made after the brain signal has already been decoded.

# ## Conclusion
# 
# You have successfully built a cost-effective, educational fMRI reconstruction pipeline. The results are not quite SOTA level but considering the amount of resources and simplicity of the methods used the results are generally promising. That being said even in SOTA pipelines not all results are perfect. Similarly in ours some reconstructions are better than others. Still I would like to congratulate you on coming this far!
# 
# The key insight from this tutorial series is that reconstruction works best when we leverage multiple levels of visual representation. The VAE latents from Notebook 3 capture low-level spatial information that the visual cortex encodes in early processing stages. The CLIP embeddings from Notebook 4 capture high-level semantic information that emerges in later visual areas. By combining both, we can produce reconstructions that are both spatially coherent and semantically meaningful.
# 
# This hybrid approach reflects the structure of a popular model of the visual system itself. Early visual cortex processes edges, colors, and spatial frequencies. Higher visual areas respond to objects, faces, scenes, and abstract categories. A complete model of visual perception needs to capture both levels, and so does a complete reconstruction pipeline.
# 
# The methods we used here follow the same principles as state-of-the-art approaches like MindEye and Brain-Diffuser. Those methods use more sophisticated training procedures, larger models, and more compute, but the core idea of fusing low-level structure with high-level semantics via diffusion generation remains the same.
# 
# 

# ## Important Caveats
# 
# The results in this tutorial depend heavily on the exceptional quality of the Natural Scenes Dataset. NSD was collected using a 7-Tesla scanner with 1.8mm isotropic voxel resolution, providing much finer spatial detail than the 3mm voxels typical of clinical and research scanners. Each subject completed 30 to 40 hours of scanning across dozens of sessions, generating far more data per person than most neuroimaging studies which often have only 1 to 2 hours per participant.
# 
# The raw fMRI data also required extensive preprocessing before we could use it. Motion correction compensated for head movements during scanning. Temporal filtering removed physiological noise and scanner drift. Precise anatomical alignment ensured that voxels could be consistently identified across sessions. We abstracted away all of this complexity by using the preprocessed NSD data, but it represents a significant amount of work that would need to be replicated for new datasets.
# 
# Generalization is another fundamental challenge  NSD consists exclusively of natural scenes drawn from the COCO dataset. How well these models would perform on other image domains and datasets, especially novel ones, is not well-defined.

# ## Further Reading and Current Research
# 
# The field of fMRI reconstruction is moving rapidly. This notebook is heavily inspired by pioneering 2023 works such as **MindEye** (Scotti et al.) and **Brain-Diffuser** (Ozcelik & VanRullen), both of which established the paradigm of aligning brain signals to multimodal CLIP spaces and using diffusion models for generation.
# 
# As of 2024 and 2025, the cutting edge has continued to advance. Newer methods like **MindEye2** (Scotti et al., 2024) have scaled these approaches to larger backbones and introduced **Shared-Subject Models**, radically reducing the amount of training data needed. Similarly, novel architectures like **Brain-IT** (Beliy et al.) are refining how we model the functional interaction between brain regions to strictly improve the fidelity of **natural scene reconstruction**, moving beyond simple pixel alignment to capture complex visual relationships.
# 
# ### References
# 
# 1.  **MindEye:** Scotti, P. S., Banerjee, A., Goode, J., Shabalin, S., Nguyen, A., Cohen, E., ... & Abraham, T. M. (2023). MindEye: fMRI-to-Image Reconstruction with Retrieval Augmented Diffusion. *arXiv preprint arXiv:2305.10865*.
# 2.  **Brain-Diffuser:** Ozcelik, F., & VanRullen, R. (2023). Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion. *arXiv preprint arXiv:2303.05334*.
# 3.  **MindEye2:** Scotti, P. S., et al. (2024). MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. *ICML 2024*.
# 4.  **Brain-IT:** Beliy, R., Zalcher, A., Kogman, J., Wasserman, N., & Irani, M. (2025). Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer. *arXiv preprint arXiv:2510.25976*.
# 
# Thank you for following along with this tutorial series! We hope this tutorial series provided a gentle introduction to natural stimulus reconstruction!
