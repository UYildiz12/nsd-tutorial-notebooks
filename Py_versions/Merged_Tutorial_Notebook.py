#!/usr/bin/env python
# coding: utf-8

# # Full NSD Tutorial (Merged)
# 
# This merged export concatenates the current tutorial notebooks in run order so it stays aligned with the canonical CurrentTutorialPipeline notebooks.

# ## Source: Notebook0_Basics

# # Notebook 0: Colab Basics
# 
# This notebook covers the essentials for running the tutorial series: executing cells, managing files, installing packages, and enabling GPU acceleration. If you're comfortable with Colab, skip ahead to Notebook 1.

# ## What You'll Build
# 
# By the end of this tutorial series, you will reconstruct the images a person was viewing **directly from their brain activity**. We'll take fMRI voxel patterns and decode them into photorealistic images using modern deep learning.
# 
# | Notebook | Topic | What You'll Learn |
# |----------|-------|-------------------|
# | **0 (This one)** | Colab Basics | Environment setup, GPU access |
# | **1** | fMRI & Voxels | What brain data looks like, loading NSD |
# | **2** | VAEs | Image compression, latent spaces |
# | **3** | Low-Level Pipeline | Predicting spatial structure from brain activity |
# | **4** | High-Level Pipeline | Predicting semantic content (CLIP embeddings) |
# | **5** | Hybrid Reconstruction | Combining both for final image reconstruction |
# 
# **Estimated total runtime**: 1-1.5 hours

# 
# This notebook is purely setup and basics. Its goals are:
# 
# 
# - Demonstrate the minimum Colab workflow (cells, files, runtime).
# - Show how to work with python packages, install them and use libraries.
# - Turn on a GPU and verify PyTorch can use it.
# 
# Without further ado lets start!

# ### Colab workflow in a rush
# 
# Colab notebooks are just a sequence of cells.
# 
# - **Run a cell**: click the play button or press `Shift + Enter`.
# - **Restart** if things get weird: `Runtime → Restart runtime`.
# - **Change runtime** (CPU/GPU): `Runtime → Change runtime type`.
# - **Files** live in the left sidebar (folder icon).
#   - Anything under `/content` is temporary and will be deleted when the session ends.
#   - We need persistance because of our pipelines modular nature. We will use Google Drive to achieve this.
# 
# So far what we had was only markdown cells, basically text. From now on you're going to run code cells.
# 
# One advantage of Colab is that code cells can be run in isolation and out of order. This allows you to fix a bug in the middle of a script and re-run just that part without restarting the entire process.
# 
# However, keep in mind:
# 
# Global State: Variables are stored in memory once a cell is run. If you define x = 10 in Cell A, run it, change it to x = 20, and run it again, the current value is 20, even if you jump down to Cell Z.
# 
# The "Hidden State" Trap: If you run cells out of order (e.g., Cell 4, then Cell 2), your code might work now but fail when run from top-to-bottom later. Rule of thumb: Before finishing, always do Runtime → Restart and run all to verify reproducible results.
# 
# We will demonstrate this below:

# In[ ]:


# Run this cell once
my_number = 3
print(f"Your number: {my_number}")


# In[ ]:


# Run this cell multiple times (Ctrl + Enter)
my_number += 1
print(f"1 + My number is: {my_number}")


# From this example it might not seem that dangerous but Colab environments get complicated quick, especially when you are trying to debug an issue and try solutions multiple times. We ourselves suffered from this issue many times.

# By default, the files you see in the left sidebar (the folder icon) are temporary. They live on the virtual machine that Google lent you for this session. If your runtime disconnects or restarts, all files in /content are deleted.

# One easy way to counter this is using your Drive as a consistent storage for colab. In the following cell, you can test this. Mounting your drive to this or any other notebook in this series will **not** share your data with anyone.

# In[ ]:


from google.colab import drive

# This will trigger an authentication prompt
drive.mount('/content/drive')


# In the following cell we create a temporary txt file

# In[ ]:


import os

# 1. Create a dummy file in the temporary Colab storage
with open('colab_test.txt', 'w') as f:
    f.write("This file was created in Colab and saved to Drive!")


# And save it to your drive:

# In[ ]:


get_ipython().system('cp colab_test.txt /content/drive/MyDrive/')


# Then we check if it is there:

# In[ ]:


if os.path.exists('/content/drive/MyDrive/colab_test.txt'):
    with open('/content/drive/MyDrive/colab_test.txt', 'r') as f:
        print("File is there. Its content is:", f.read())


# And delete it again so we don't treat it like garbage

# In[ ]:


# Remove the test file from your Google Drive
get_ipython().system('rm /content/drive/MyDrive/colab_test.txt')

print("Cleaned up! 'colab_test.txt' has been removed from your Drive.")


# Check again:

# In[ ]:


import os

if os.path.exists('/content/drive/MyDrive/colab_test.txt'):
    print("File still exists.")
else:
    print("File successfully deleted from Drive.")


# Colab is popular because it comes with "batteries included" most heavy-lifting machine learning libraries are already there. But for neuroscience especially you might need to install some more.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"NumPy version: {np.__version__}")
print("Standard libraries are already here.")


# In[ ]:


# Try to import a library that isn't pre-installed
try:
    import emoji
except ImportError:
    print("Library not found. Installing now...")
    # The '!' tells Colab this is a terminal command, not Python
    # -q quiets the output logs
    get_ipython().system('pip install emoji -q')
    import emoji

print(emoji.emojize("Wow this notebook is :fire:"))


# For these notebooks, you will need GPU access. Luckily, Google Colab provides this based on dynamic availability.
# 
# If you have Colab Pro or have a few compute units in your account, this tutorial series should run without a hitch. If not, you may need to wait for T4 availability, particularly for the encoding, decoding, and training notebooks.
# 
# We have tried our best to make these notebooks as beginner-friendly as possible, but due to the nature of deep learning, adequate compute resources are still key. With that being said you can check if the current session has a cuda supporting notebook by running the following cell.

# In[ ]:


import torch
if torch.cuda.is_available():
    print(f"Success! GPU Detected: {torch.cuda.get_device_name(0)}")
    print("PyTorch will run fast.")
else:
    print("No GPU detected. Did you change the Runtime type?")
    print("PyTorch will run slowly on the CPU and might crash unexpectedly.")


# ## Next Steps
# 
# You're all set! You learned the basics of colab and are ready to use it to your advantage!
# 
# **Continue to [Notebook 1: fMRI & Voxels](https://colab.research.google.com/drive/1J1KX6V6ZGwjbcZ65XPD8vfkCqrtnPoXG#scrollTo=UTiFdDsg7HHi)** where we'll learn what fMRI actually measures, load our dataset and take a look at the activation patterns.
#

# ## Source: Notebook1_fMRI

# # Notebook 1: fMRI & Voxels

# Before we dive in, it's worth acknowledging that these tutorials are designed for people with limited experience in decoding. With that in mind, you might find yourself asking questions like:
# 
# 
# 
# *   **How is this even possible?**
# *   **What exactly are we predicting?**
# *   **What does “reconstruction” actually mean here?**  
# 
# By the end of this tutorial, the answers to these questions (and many others) should become intuitively clear. However, rather than addressing all of them upfront, I want to tackle the first one directly and make a few subsequent points.
# 

# # How is this possible?
# 
# At its core, fMRI-based visual reconstruction rests on a deceptively simple idea: when you look at an image, your brain produces patterns of activity that systematically relate to the visual features in that image and these patterns are consistent enough to be learned and reversed.

# ## The Encoding-Decoding Framework
# 
# Think of the brain as an **encoder**. It takes a high-dimensional image and transforms it into a pattern of neural activity.
# 
# $$\text{Image} \xrightarrow{\text{Brain (Encoder)}} \text{Neural Activity}$$
# 
# Our job is to build a **decoder**, a model that reverses this process:
# 
# $$\text{Neural Activity} \xrightarrow{\text{Our Model (Decoder)}} \text{Reconstructed Image}$$
# 
# This is what we mean by "visual reconstruction" or "decoding." We're not reading minds in any mystical sense—we're learning a mathematical mapping from brain patterns back to images. The brain, of course, does the encoding for free, we just need to learn to undo it.

# # What fMRI Actually Measures
# 
# Functional Magnetic Resonance Imaging (fMRI) doesn't directly measure neural activity. Instead, it measures changes in blood oxygenation levels across the brain—what's known as the BOLD signal (Blood Oxygen Level Dependent). When neurons in a particular region become active, they consume oxygen, triggering increased blood flow to that area. fMRI detects these changes.
# 
# fMRI divides the brain into a 3D grid of tiny cubes called **voxels** (volumetric pixels). Each voxel is typically 2-3mm on each side and contains hundreds of thousands of neurons. For every voxel, fMRI gives us a single number representing the BOLD signal at that location.
# 
# This means a brain scan is essentially a long vector of numbers:
# 
# $$b = [b_1, b_2, b_3, ..., b_n]$$
# 
# Where:
# - $b_i$ = the BOLD signal in voxel $i$
# - $n$ = total number of voxels
# 
# When we talk about "brain activity" throughout this tutorial, we're really talking about this vector $b$ a high-dimensional snapshot of blood oxygenation levels across the brain at a given moment.

# ## Working with NSD Data
# 
# Now that we understand what fMRI measures, let's get our hands on some real data. We'll be using the **Natural Scenes Dataset (NSD)**—one of the largest and most comprehensive fMRI datasets available for visual neuroscience.
# 
# ### What is NSD?
# 
# The Natural Scenes Dataset is a massive publicly available dataset where 8 subjects viewed thousands of natural images while undergoing high-resolution fMRI scanning.
# 
# ### Why NSD?
# 
# NSD has become the benchmark dataset for visual reconstruction research for several reasons:
# 
# 1. **Scale** — Thousands of image-brain pairs per subject (deep learning loves data)
# 2. **Quality** — 7 Tesla scanning provides exceptional signal quality
# 3. **Standardization** — Preprocessed and organized, ready for machine learning
# 4. **Natural images** — Complex, real-world scenes rather than simple stimuli
# 
# With that being said lets load it up!

# In[ ]:


# Install dependencies
get_ipython().system('pip install -q nibabel numpy matplotlib webdataset braceexpand nilearn')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
import braceexpand
import nilearn
from PIL import Image


# In[ ]:


def load_nsd_subset(subject_id=1, shards="0..5", max_samples=500):
    """Load a subset of NSD training data."""

    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_new"
    url_pattern = f"{base_url}/train/train_subj0{subject_id}_{{{shards}}}.tar"
    urls = list(braceexpand.braceexpand(url_pattern))

    print(f" Loading NSD Subject {subject_id}")
    print(f" Shards: {shards} ({len(urls)} files)")

    dataset = wds.WebDataset(urls)

    voxels_list = []
    images_list = []

    for sample in dataset:
        # Get voxels with the nsdgeneral.npy mask
        voxels = np.load(io.BytesIO(sample['nsdgeneral.npy']))

        # Get image
        img = Image.open(io.BytesIO(sample['jpg'])).convert('RGB')
        image = np.array(img) / 255.0

        voxels_list.append(voxels)
        images_list.append(image)

        if len(voxels_list) % 100 == 0:
            print(f"   Loaded {len(voxels_list)} samples...")

        if len(voxels_list) >= max_samples:
            break

    voxels = np.stack(voxels_list)
    images = np.stack(images_list)

    print(f"   Loaded {len(voxels)} samples")
    print(f"   Voxels: {voxels.shape}")
    print(f"   Images: {images.shape}")

    return voxels, images


# In[ ]:


# Load data
voxels, images = load_nsd_subset(subject_id=1, shards="0..5", max_samples=500) #500 Images only for now.


# In[ ]:


trial_a, trial_b = 0, 25
voxel_range = slice(0, 300)

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

axes[0, 0].imshow(images[trial_a])
axes[0, 0].set_title(f'Trial {trial_a}')
axes[0, 0].axis('off')

axes[0, 1].imshow(images[trial_b])
axes[0, 1].set_title(f'Trial {trial_b}')
axes[0, 1].axis('off')

axes[1, 0].plot(voxels[trial_a, 0, voxel_range], color='C0', alpha=0.7)
axes[1, 0].set_ylabel('BOLD Signal')
axes[1, 0].set_xlabel('Voxel Index')

axes[1, 1].plot(voxels[trial_b, 0, voxel_range], color='C1', alpha=0.7)
axes[1, 1].set_xlabel('Voxel Index')

plt.suptitle('Different images produce different brain activation patterns', fontsize=13)
plt.tight_layout()
plt.show()


# ## Exploring the Data
# 
# Before diving into the theory, let's get our hands dirty and explore what we just loaded. Understanding the shape and characteristics of your data is always the first step.

# ### Why 3 Repetitions?
# 
# You may notice our voxel data has shape `(samples, 3, 15724)`. Each image in NSD was shown to the subject **three times** across different scanning sessions. The main reason for this is that single-trial fMRI data suffers from an extremely low signal-to-noise ratio, capturing significant fluctuations from physiological processes and background neural activity not immediately relevant to the visual task. To accurately determine how a stimulus is represented in the brain, multiple repetitions are effectively essential.
# 
# ### How should you use them?
# 
# You have two main options. The first is to average across the three repetitions for each image. Since the true brain response to an image stays roughly the same each time while the noise fluctuates randomly, averaging cancels out much of the noise and gives you a cleaner signal. The tradeoff is that this reduces your total number of samples. The second option is to treat each repetition as its own training sample, which gives you 3x more training data but with noisier individual samples.
# 
# Averaging works well when you want the most faithful representation of how the brain responds to each image. Keeping them separate tends to benefit deep learning models that are data-hungry and can learn to handle noise on their own. The best choice depends on your model architecture and training strategy, and the webdataset provides all three repetitions so you're free to go either way.

# In[ ]:


# Visualize the effect of averaging repetitions
trial = 10

# Plot 1: Overlay all three repetitions to show variability
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Show a slice of voxels across repetitions so you can see the fluctuation
voxel_range = slice(0, 200)

for rep in range(3):
    axes[0].plot(voxels[trial, rep, voxel_range], alpha=0.6, label=f'Rep {rep+1}')
axes[0].set_title('Three Repetitions of the Same Image')
axes[0].set_xlabel('Voxel Index')
axes[0].set_ylabel('BOLD Signal')
axes[0].legend()

# Show averaged signal on top of a single rep for direct comparison
axes[1].plot(voxels[trial, 0, voxel_range], alpha=0.4, color='coral', label='Single Rep')
axes[1].plot(voxels[trial].mean(axis=0)[voxel_range], color='seagreen', linewidth=2, label='Averaged')
axes[1].set_title('Single Rep vs Averaged Signal')
axes[1].set_xlabel('Voxel Index')
axes[1].set_ylabel('BOLD Signal')
axes[1].legend()

plt.suptitle(f'Trial {trial}: Noise Reduction Through Averaging', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Quantify the difference
single_std = voxels[trial, 0, :].std()
averaged_std = voxels[trial].mean(axis=0).std()
print(f"Std of single repetition: {single_std:.4f}")
print(f"Std of averaged signal:   {averaged_std:.4f}")
print(f"Averaging reduced variance by {(1 - averaged_std/single_std) * 100:.1f}%")


# ### The nsdgeneral Mask: Where Vision Lives
# 
# When we load voxel data from NSD, we're not loading the *entire* brain. Instead, we focus on the **nsdgeneral** masked voxels.
# 
# ### What is nsdgeneral?
# 
# The `nsdgeneral` mask is a binary ROI (Region of Interest) that spans the visually-responsive cortex. It includes early visual areas (V1, V2, V3) where basic features like edges are processed, higher visual areas (V4, LO, IT) where complex shapes emerge, and specialized regions for faces and places (FFA, PPA).
# 
# This mask contains approximately **15,724 voxels** per subject, carefully selected because they show reliable responses to visual stimuli.
# 
# ### How Binary Masking Works
# 
# We call this a "binary" mask because it operates on a strict **Keep (1) or Discard (0)** logic.
# 
# Think of the mask as a rigid filter applied to the 3D brain volume:
# * **1 (Keep):** If a voxel is inside the visual cortex, we retain its data.
# * **0 (Discard):** If a voxel is outside these regions, we ignore it entirely.
# 
# Unlike other masking techniques that might "softly" weight regions by importance, a binary mask completely removes data from non-visual areas. We  are effectively cutting voxels out of our dataset to ensure we only analyze the signal relevant to vision.
# 
# ### Why Use a Mask At All?
# 
# $$\text{Full brain} \approx 200,000 \text{ voxels}$$
# $$\text{nsdgeneral} \approx 15,724 \text{ voxels}$$
# 
# By focusing on visually-responsive regions, we reduce noise from irrelevant brain areas, improve computational efficiency, and boost the signal that actually encodes visual information. The voxel vector we loaded earlier comes pre-masked to `nsdgeneral`, so each sample is already optimized for visual decoding.

# In[ ]:


import urllib.request
import nibabel as nib

os.makedirs('nsd_data', exist_ok=True)

subject = 1
subj_str = f'subj0{subject}'
base_url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata"

# ROI files
roi_files = {
    'nsdgeneral': f'{base_url}/ppdata/{subj_str}/func1pt8mm/roi/nsdgeneral.nii.gz',
    'visual_rois': f'{base_url}/ppdata/{subj_str}/func1pt8mm/roi/prf-visualrois.nii.gz',
    'face_rois': f'{base_url}/ppdata/{subj_str}/func1pt8mm/roi/floc-faces.nii.gz',
    'place_rois': f'{base_url}/ppdata/{subj_str}/func1pt8mm/roi/floc-places.nii.gz',
}

# Anatomical reference
anat_files = {
    'T1': f'{base_url}/ppdata/{subj_str}/func1pt8mm/T1_to_func1pt8mm.nii.gz',
}

print(f" Downloading NSD data for Subject {subject}...")

for name, url in {**roi_files, **anat_files}.items():
    filepath = f'nsd_data/{name}.nii.gz'
    if not os.path.exists(filepath):
        try:
            print(f"   Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"    Failed: {e}")
    else:
        print(f"   {name} exists")


# ## Visualizing Brain Activity
# 
# Now that we have our voxel data (a 1D vector) and the nsdgeneral mask (a 3D binary volume), we need to **map them back together** for visualization.
# 
# ### The Mapping Process
# 
# The webdataset provides voxels as a flattened array:
# $$\text{voxels} = [v_1, v_2, v_3, ..., v_{15724}]$$
# 
# The nsdgeneral mask tells us *where* in 3D space each voxel belongs. We load the mask, find the 3D coordinates where `mask > 0`, and insert each voxel value at its corresponding location. This gives us a 3D brain volume we can visualize alongside the T1 anatomical image.

# In[ ]:


# Clean NSD Brain Visualization with nilearn
from nilearn import plotting
import nibabel as nib

# Load data
t1_img = nib.load('nsd_data/T1.nii.gz')
roi_img = nib.load('nsd_data/nsdgeneral.nii.gz')
roi_data = roi_img.get_fdata()

# Pick a trial
trial_idx = np.random.randint(len(images))
trial_voxels = voxels[trial_idx]
if trial_voxels.ndim > 1:
    trial_voxels = trial_voxels.mean(axis=0)

# Map to 3D volume
roi_indices = np.where(roi_data > 0)
activation_volume = np.zeros_like(roi_data, dtype=np.float32)
activation_volume[roi_indices] = trial_voxels.astype(np.float32)
activation_nii = nib.Nifti1Image(activation_volume, roi_img.affine, roi_img.header)

vmax = float(np.percentile(np.abs(trial_voxels), 98))

# Show stimulus
print(f" Stimulus Image #{trial_idx}")
plt.figure(figsize=(5, 5))
plt.imshow(images[trial_idx])
plt.axis('off')
plt.title(f'Image #{trial_idx}', fontsize=12)
plt.show()

# Show brain activation
plotting.view_img(
    activation_nii,
    bg_img=t1_img,
    threshold=0.2,
    cmap='cold_hot',
    symmetric_cmap=True,
    vmax=vmax,
    title=f'Brain Activity #{trial_idx}'
)


# ## The Reconstruction Challenge
# 
# Visual reconstruction is fundamentally an **inverse problem**. The brain transforms a high-dimensional image into a pattern of neural activity. Reconstruction asks us to reverse this process.
# 
# Consider the dimensionality mismatch: we have ~15,000 voxels trying to predict an image with ~200,000 pixels (256×256 RGB). This is severely underdetermined. There are infinitely many images that could produce similar brain activity.
# 
# What makes this tractable? The space of *natural*  images is highly structured. Modern generative models have learned these structures, and by mapping brain activity into their latent spaces, we can leverage this prior knowledge to resolve the ambiguity.
# 
# In the next notebook, we'll explore **Variational Autoencoders (VAEs)** which are models that compress images into compact latent representations, giving us a more tractable target to predict from brain activity. We'll continue with our [Notebook 2: Variational Autoencoders](https://colab.research.google.com/drive/1lrT398xL_wdc6Pr1-Vrgja10OWm9C1xQ)

# ## Source: Notebook2_VAEs

# # Notebook 2: Understanding VAEs for Brain Decoding
# 
# Before we can decode brain activity into images, we need to understand the representation we'll be predicting. This notebook introduces **Variational Autoencoders (VAEs)**—the compression backbone that makes brain-to-image reconstruction tractable.
# 
# The core problem is dimensionality. A 256×256 image has nearly 200,000 pixel values, but we only have ~8,000 training examples. Predicting pixels directly is hopeless. VAEs solve this by compressing images into a 4,096-dimensional latent space—a 48× reduction that preserves perceptually meaningful structure while discarding redundant detail.
# 
# We'll explore how VAEs encode and decode images, why their latent space is well-suited for regression, and how reconstruction quality guides the next steps. By the end, you'll understand exactly what we're asking the brain decoder to predict in the next notebook.

# In[ ]:


get_ipython().system('pip -q install diffusers transformers accelerate torch torchvision matplotlib webdataset braceexpand pillow')


# ## Key libraries
# 
# This notebook uses **PyTorch** for tensor operations and GPU compute, **diffusers** to load pre-trained VAE models from the Stable Diffusion family, and **matplotlib** for visualization.

# In[ ]:


# Standard libraries
import os
import io
from pathlib import Path

# Numeric + PyTorch
import numpy as np
import torch
import torch.nn.functional as F

# VAE from diffusers
from diffusers import AutoencoderKL

# Dataset streaming
import webdataset as wds
import braceexpand
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Set seed for reproducibility
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ## Part 1: What is a Variational Autoencoder (VAE)?
# 
# A **Variational Autoencoder** is a neural network that learns to compress and reconstruct data. It consists of three components that work together: an encoder, a latent space, and a decoder.
# 
# The **encoder** takes an input image and compresses it into a latent vector. Think of it as summarizing an image into its most essential features.
# 
# Mathematically, $\text{Encoder}: \mathbf{x} \rightarrow (\boldsymbol{\mu}, \boldsymbol{\sigma})$. The encoder outputs a *distribution* over latents (mean $\mu$ and standard deviation $\sigma$), not just a single point.
# 
# The **latent space** is the compressed representation where similar images map to nearby points. For the SD-VAE we use here, a `256×256×3` input image (196,608 values) becomes a `32×32×4` latent tensor (4,096 values)—a **48× compression**. The latent space preserves perceptually important features while discarding redundant pixel-level details.
# 
# The **decoder** reconstructs an image from a latent vector: $\text{Decoder}: \mathbf{z} \rightarrow \hat{\mathbf{x}}$. The reconstruction $\hat{\mathbf{x}}$ should look like the original $\mathbf{x}$, but it won't be pixel-perfect.
# 
# ```
# Image (256×256×3) → [Encoder] → Latent (32×32×4) → [Decoder] → Reconstruction (256×256×3)
#    196,608 values              4,096 values                   196,608 values
# ```
# 
# The VAE is trained to minimize reconstruction error while keeping the latent space well-organized (regularized). This balance is what makes VAEs useful for generative modeling.

# In[ ]:


# Configuration
VAE_ID = "stabilityai/sd-vae-ft-mse"  # Same VAE as Tutorial 1
IMG_SIZE = 256

# Paths
PROJECT_DIR = Path.cwd().parent  # Go up from Notebooks/ to project root
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"VAE: {VAE_ID}")
print(f"Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"Cache directory: {CACHE_DIR}")


# In[ ]:


# Load the Stable Diffusion VAE
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).eval()

# Print architecture summary
print(f"\nVAE Architecture:")
print(f"  Scaling factor: {vae.config.scaling_factor}")
print(f"  Latent channels: {vae.config.latent_channels}")
print(f"  Input: {IMG_SIZE}×{IMG_SIZE}×3 → Latent: {IMG_SIZE//8}×{IMG_SIZE//8}×{vae.config.latent_channels}")

latent_dim = (IMG_SIZE // 8) * (IMG_SIZE // 8) * vae.config.latent_channels
pixel_dim = IMG_SIZE * IMG_SIZE * 3
print(f"  Compression: {pixel_dim:,} pixels → {latent_dim:,} latents ({pixel_dim/latent_dim:.1f}× reduction)")


# ## Part 2: Why Use SD-VAE Latents for Brain Decoding?
# 
# In natural stimulus reconstruction, our ultimate goal is to create a faithful reconstruction of the ground truth image (the image the participant saw while they were in the fMRI machine). So why even bother with some compression algorithm? Why not just predict the pixels directly?
# 
# The first problem is **dimensionality**. A `256×256×3` RGB image has **196,608 dimensions**. In the NSD dataset, we have approximately ~8,000 training samples (images shown to a subject) and ~15,000 voxels (fMRI measurements per image). This means we're trying to learn a mapping from 15,000 inputs to 196,608 outputs with only 8,000 examples. This is **severely underconstrained**—we have far more parameters to learn than training samples. Regularization (like Ridge regression) helps, but you're fundamentally fighting an uphill battle. Imagine trying to predict 200,000 numbers from 15,000 numbers, but you only have 8,000 examples to learn from. For every output dimension, you need to learn which input voxels matter and how much. With so few examples, the model will overfit to noise rather than learning meaningful patterns.
# 
# The second problem is that **pixel space is perceptually harsh**. Mean Squared Error (MSE) in pixel space penalizes tiny spatial shifts severely. A reconstruction that's shifted by just 1 pixel looks **fine to humans** but has **huge MSE** because every pixel is technically "wrong." Meanwhile, an image with completely wrong colors but perfect alignment might have lower MSE despite looking terrible. This mismatch between perceptual quality and mathematical loss makes pixel-space regression frustrating—your loss can be high even when reconstructions look good.
# 
# 
# ![image.png](data:image/png;base64,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)
# 
# 
# 
# The solution is to predict **VAE latents** instead. By predicting latents (4,096 dimensions) instead of pixels (196,608 dimensions), we reduce the target dimensionality by 48×, going from 0.04 samples per dimension to nearly 2 samples per dimension. The latent space is also **perceptually organized**: images that look similar to humans have similar latent representations. This means small errors in latent space correspond to small perceptual errors, unlike pixel space where small spatial shifts cause large losses. Additionally, latents have partial spatial invariance which allows small shifts in the image to get absorbed rather than cause catastrophic loss increases.
# 
# Mathematically, in pixel-space regression we would learn $f_\theta: \mathbf{v} \rightarrow \hat{\mathbf{x}} \in \mathbb{R}^{196608}$, but in latent-space regression we learn $g_\phi: \mathbf{v} \rightarrow \hat{\mathbf{z}} \in \mathbb{R}^{4096}$ and then apply the pre-trained decoder: $\hat{\mathbf{x}} = \text{VAE.decode}(\hat{\mathbf{z}})$. The regression problem $g_\phi$ is **48× easier** than $f_\theta$ because we're predicting far fewer values, and each latent dimension captures meaningful information.
# 
# The trade-off is that your reconstructions must pass through the VAE decoder. This means that even with perfect latent prediction, $\text{VAE.decode}(\text{VAE.encode}(\mathbf{x})) \neq \mathbf{x}$. In practice however, the difference is largely indistinguishable as we'll see in Part 4, and more than good enough for our purposes in most cases.

# ## Part 3: Loading NSD Images
# 
# Before we demonstrate the VAE, we need real images. We'll use the Natural Scenes Dataset (NSD), streaming a small subset directly from HuggingFace. This is the same dataset used in Tutorial 1 for brain decoding.

# In[ ]:


# Helper functions for encoding and decoding

def to_vae_range(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images to VAE input range [-1, 1].
    Handles both uint8 [0,255] and float [0,1] inputs.
    """
    x = images.float()
    if x.max() > 1.5:  # uint8 input
        x = x / 255.0
    x = x.clamp(0, 1)
    x = x * 2 - 1  # [0,1] -> [-1,1]
    return x


@torch.inference_mode()
def encode_to_latents(images: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """
    Encode images to scaled SD-VAE latents.

    Args:
        images: [B, 3, H, W] tensor in [0,1] or [0,255]
        vae: Loaded AutoencoderKL model

    Returns:
        latents: [B, 4, H/8, W/8] scaled latents
    """
    scaling_factor = float(vae.config.scaling_factor)

    x = to_vae_range(images).to(device)

    # Resize if needed
    if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

    # Encode to latent distribution and take the mode (deterministic)
    latent_dist = vae.encode(x).latent_dist
    latents = latent_dist.mode() * scaling_factor

    return latents.cpu()


@torch.inference_mode()
def decode_from_latents(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """
    Decode scaled SD-VAE latents back to images.

    Args:
        latents: [B, 4, H/8, W/8] scaled latents
        vae: Loaded AutoencoderKL model

    Returns:
        images: [B, 3, H, W] in [0, 1]
    """
    scaling_factor = float(vae.config.scaling_factor)

    z = latents.to(device) / scaling_factor
    x = vae.decode(z).sample
    x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

    return x.cpu()


print("Encoding/decoding functions defined.")


# In[ ]:


# Load NSD images via streaming (same approach as Tutorial 1)

def load_nsd_images(subject_id: int = 1, max_samples: int = 100):
    """
    Stream NSD images from HuggingFace.
    Returns images as a tensor [N, 3, H, W] in [0, 1] range.
    """
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_new"
    url_pattern = f"{base_url}/train/train_subj0{subject_id}_{{0..2}}.tar"  # First 3 shards
    urls = list(braceexpand.braceexpand(url_pattern))

    print(f"Loading NSD images for Subject {subject_id}...")
    dataset = wds.WebDataset(urls)

    images_list = []
    for sample in dataset:
        # Decode image
        img = Image.open(io.BytesIO(sample['jpg'])).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        images_list.append(img_tensor)

        if len(images_list) % 25 == 0:
            print(f"  Loaded {len(images_list)} images...")

        if len(images_list) >= max_samples:
            break

    images = torch.stack(images_list)
    print(f"  Done! Loaded {len(images)} images, shape: {images.shape}")
    return images


# Load a small batch for demonstration
nsd_images = load_nsd_images(subject_id=1, max_samples=50)
print(f"\nNSD images shape: {nsd_images.shape}")
print(f"Value range: [{nsd_images.min():.3f}, {nsd_images.max():.3f}]")


# ## Part 4: Seeing the VAE in Action
# 
# Now that we have our NSD images loaded, let's see how the VAE actually performs. We'll pass images through the encoder to get latents, then decode those latents back into images. This round-trip will show us exactly what information the VAE preserves and what it discards, which directly determines the brain decoder's performance.

# In[ ]:


# Encode and decode the first few NSD images
test_images = nsd_images[:4]  # Use first 4 for visualization

print("Encoding images to latents...")
latents = encode_to_latents(test_images, vae)
print(f"Latent shape: {latents.shape}")
print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")

print("\nDecoding latents back to images...")
reconstructions = decode_from_latents(latents, vae)
print(f"Reconstruction shape: {reconstructions.shape}")


# In[ ]:


# Visualize: Original vs Reconstruction
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Resize originals to match reconstruction size for fair comparison
test_images_resized = F.interpolate(test_images, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

for i in range(4):
    # Original (show resized version for consistency)
    axes[0, i].imshow(test_images_resized[i].permute(1, 2, 0).clamp(0, 1))
    axes[0, i].set_title(f"NSD Image #{i+1}")
    axes[0, i].axis('off')

    # Reconstruction
    axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).clamp(0, 1))
    mse = F.mse_loss(test_images_resized[i], reconstructions[i]).item()
    axes[1, i].set_title(f"VAE Recon (MSE={mse:.4f})")
    axes[1, i].axis('off')

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("VAE Reconstruction", fontsize=12)

plt.suptitle("VAE Encode → Decode on Real NSD Images", fontsize=14)
plt.tight_layout()
plt.show()


# So right now the trade-off might not exactly be looking like a trade-off. The VAE reconstructions are almost indistguishable. But some might argue that these images were in the VAEs training data the NSD being a subset of COCO images and all. We can try an entirely synthetic image and take a look at whether it works.

# In[ ]:


# Create a synthetic test image the VAE has NEVER seen
def create_novel_test_image(size=256):
    """Create a synthetic image with patterns the VAE never saw during training."""
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Gradient background
    for i in range(size):
        img[i, :, 0] = i / size  # Red gradient top to bottom
        img[:, i, 2] = i / size  # Blue gradient left to right

    # Add geometric shapes
    # Circle
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 4
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2
    img[circle_mask] = [0, 1, 0]  # Green circle

    # Smaller circle inside
    inner_radius = size // 8
    inner_mask = (x - center)**2 + (y - center)**2 <= inner_radius**2
    img[inner_mask] = [1, 1, 0]  # Yellow inner circle

    # Add some stripes in corner
    for i in range(0, size//4, 8):
        img[i:i+4, :size//4, :] = [1, 0, 1]  # Magenta stripes

    # Add text-like pattern (checkerboard in corner)
    checker_size = 8
    for i in range(size//4, size//2):
        for j in range(size//4, size//2):
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                img[i, j] = [1, 1, 1]  # White

    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


# Create and process our novel image
novel_image = create_novel_test_image(IMG_SIZE)
print(f"Novel image shape: {novel_image.shape}")

# Encode and decode
novel_latent = encode_to_latents(novel_image, vae)
novel_recon = decode_from_latents(novel_latent, vae)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

axes[0].imshow(novel_image[0].permute(1, 2, 0).clamp(0, 1))
axes[0].set_title("Original Synthetic Image\n(VAE never saw this)")
axes[0].axis('off')

axes[1].imshow(novel_recon[0].permute(1, 2, 0).clamp(0, 1))
mse = F.mse_loss(novel_image, novel_recon).item()
axes[1].set_title(f"VAE Reconstruction\n(MSE={mse:.4f})")
axes[1].axis('off')

# Difference map
diff = (novel_image - novel_recon).abs()
axes[2].imshow(diff[0].permute(1, 2, 0).clamp(0, 1) * 3)  # Amplify for visibility
axes[2].set_title("Difference (3× amplified)")
axes[2].axis('off')

plt.suptitle("VAE Round-Trip on Novel Image (Never in Training)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# The VAE reconstructs this synthetic image reasonably well,
# proving it learned general compression principles, not just memorized training data.
# Notice where it struggles: sharp edges, fine patterns, and exact color preservation. Still it is more than enough for our purposes.

# 
# So then how do we use these VAEs in our reconstruction pipeline? The idea is simple enough. As you see in the round trip we can both encode images, and decode the latents that come from those images.
# 
# During **training**, we encode all stimulus images to get their latent representations. These latents become our regression targets—what we train the brain decoder to predict from fMRI voxels.
# 
# During **inference**, the brain decoder predicts latents from new fMRI data, and we pass those predicted latents through the VAE decoder to get reconstructed images:
# 
# ```
# Training:   Stimulus Image → VAE.encode() → Latent (target for regression)
#                                                  ↑
#                               Brain Decoder learns: Voxels → Latent
# 
# Inference:  Voxels → Brain Decoder → Predicted Latent → VAE.decode() → Reconstructed Image
# ```

# ## Part 5: Batch Encoding for Larger Datasets
# 
# For larger datasets, we need to encode images in batches to avoid running out of GPU memory. These batch functions will be used to encode all NSD images and compute normalization statistics.

# In[ ]:


# Batch encoding function (for larger datasets)

@torch.inference_mode()
def encode_images_batched(
    images: torch.Tensor,
    vae: AutoencoderKL,
    batch_size: int = 16
) -> torch.Tensor:
    """
    Encode images to latents in batches (memory efficient).

    Args:
        images: [N, 3, H, W] tensor
        vae: Loaded VAE model
        batch_size: Batch size for encoding

    Returns:
        latents: [N, 4, H/8, W/8] scaled latents
    """
    scaling_factor = float(vae.config.scaling_factor)
    all_latents = []

    for i in tqdm(range(0, len(images), batch_size), desc="Encoding"):
        batch = images[i:i+batch_size]
        x = to_vae_range(batch).to(device)

        if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        latent_dist = vae.encode(x).latent_dist
        latents = latent_dist.mode() * scaling_factor
        all_latents.append(latents.cpu())

        torch.cuda.empty_cache()

    return torch.cat(all_latents, dim=0)


@torch.inference_mode()
def decode_latents_batched(
    latents: torch.Tensor,
    vae: AutoencoderKL,
    batch_size: int = 16
) -> torch.Tensor:
    """
    Decode latents to images in batches (memory efficient).
    """
    scaling_factor = float(vae.config.scaling_factor)
    all_images = []

    for i in tqdm(range(0, len(latents), batch_size), desc="Decoding"):
        batch = latents[i:i+batch_size]
        z = batch.to(device) / scaling_factor
        x = vae.decode(z).sample
        x = (x.clamp(-1, 1) + 1) / 2
        all_images.append(x.cpu())

        torch.cuda.empty_cache()

    return torch.cat(all_images, dim=0)


print("Batch encoding/decoding functions defined.")


# In[ ]:


# Encode all loaded NSD images
print(f"Encoding all {len(nsd_images)} NSD images to latents...")
sample_latents = encode_images_batched(nsd_images, vae, batch_size=8)
sample_images = nsd_images

print(f"\nLatents shape: {sample_latents.shape}")
print(f"Latent stats: mean={sample_latents.mean():.4f}, std={sample_latents.std():.4f}")

# Flatten for regression
latents_flat = sample_latents.flatten(1)
print(f"Flattened shape: {latents_flat.shape}")


# ## Summary
# 
# In this tutorial, we explored **VAE architecture**: the encoder compresses images to a latent distribution, the latent space provides a compact and perceptually meaningful representation (4,096 vs 196,608 dimensions), and the decoder reconstructs images from latents.
# 
# We discussed **why SD-VAE latents are useful** for brain decoding: they offer 48× dimensionality reduction, work in a perceptual space where similar images have similar latents, and make regression tractable with limited training samples.
# 
# Finally, we demonstrated the **practical workflow**: loading NSD images, encoding/decoding with batch functions, and visualizing the VAE's round-trip reconstruction quality on both real and synthetic images.
# 
# **Next Tutorial**: We'll use these latents as regression targets for brain decoding, comparing Ridge regression and MLP approaches creating coarse outputs from our brain data. Let's start with our [Notebook 3: Low-Level Pipeline](https://colab.research.google.com/drive/1nNDbgQF-Z1OzSTJUpr1J_kLhrnOZ3Q4s)

# ## Source: Notebook3_LowLevelPipeline

# # Disk Space and Running Time Warning
# 
# This notebook takes approximately **30-35 minutes** to run fully on a T4 GPU and requires approximately **1 GB** of free disk space for the image decoding process. Ensure your drive (Google Drive if running on Colab) has enough space before running the decoding cells to avoid interruptions.
# 
# - **Test set decoded images**: ~ 750 MB (~1,000 images)
# - **Model + latents**: ~ 50 MB
# 
# Each decoded image is ~0.75 MB (256×256 RGB). Files are saved as PyTorch tensors (.pt) in batches.

# # Notebook 3: Low-Level Image Reconstruction from fMRI
# 
# In this notebook, we build a minimal low-level decoding baseline that predicts *image appearance* from fMRI by mapping voxel patterns to a compressed visual latent space.
# 
# ## What You Will Learn
# 
# By the end of this tutorial, you'll understand how to take raw fMRI voxels and produce actual image reconstructions. The key insight is that we don't predict pixels directly—instead, we predict a compressed representation that a pre-trained image decoder can turn back into pictures.
# 
# We'll work through the complete pipeline: loading NSD data, encoding images into VAE latents, fitting two different decoders (linear and nonlinear), and evaluating how well they recover visual information from brain activity.
# 
# ## What This Tutorial Is
# 
# This is **low-level reconstruction**, we're recovering color and coarse spatial structure, not semantic content. There's no text prior, no diffusion sampling, and no semantic guidance here. Think of this as establishing a baseline: how much can we recover with relatively simple methods?
# 

# In[ ]:


get_ipython().system('pip -q install diffusers transformers accelerate pytorch_msssim kornia webdataset braceexpand')


# ## Setup and Dependencies
# 
# We'll need several libraries for this tutorial: PyTorch for modeling, Diffusers for the VAE, and specialized packages for metrics and data loading. The install cell below handles everything.

# In[ ]:


# Standard libs
import os
from pathlib import Path
from typing import Tuple, Optional

# Numeric + torch
import numpy as np
import torch

# Download + dataset streaming
import braceexpand
import webdataset as wds
from dataclasses import dataclass

# Progress bars + plotting (only if you actually plot)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ## Reproducibility
# 
# Setting random seeds ensures our results are reproducible across runs. This matters for science—you want to know that differences between models come from the models themselves, not from random initialization or data shuffling.

# In[ ]:


#Seeding is important for reproducibility
import random
import os

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# In[ ]:


from dataclasses import dataclass

@dataclass(frozen=True)
class LowLevelCfg:
    subject_id: int = 1
    seed: int = 42

    # streaming
    batch_size: int = 64
    num_workers: int = 4    # Increased to 4 for faster downloads

    # low-level target
    img_size: int = 256
    vae_id: str = "stabilityai/sd-vae-ft-mse"  # SD VAE

    # ridge - lower alpha = less shrinkage = more variance in predictions
    ridge_alpha: float = 1e5

ll = LowLevelCfg()
print(ll)


# ## Configuration
# 
# We centralize all hyperparameters in a configuration object. This makes the tutorial easy to modify and rerun with different settings.
# 
# Key parameters to understand:
# - **vae_id**: Which VAE defines the latent space we're predicting into
# - **img_size**: Resolution for decoding and evaluation (256×256)
# - **ridge_alpha**: Regularization strength for the linear baseline (higher = more regularization)

# ## Streaming Data Loading
# 
# The NSD dataset is too large to download entirely, so we use **streaming data loading** with WebDataset. This approach fetches data on-demand from HuggingFace, ensuring only the batches currently being processed are held in memory.
# 
# The `build_nsd_dataset` function creates this pipeline: it downloads shards from HuggingFace, decodes compressed data containing images, voxels, and trial IDs, batches samples for efficient processing, and shuffles training data. This lets us work with datasets that far exceed available memory.

# In[ ]:


PROJECT_DIR = Path.cwd()
RESULTS_DIR = PROJECT_DIR / "results"
CACHE_DIR = PROJECT_DIR / "cache"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"CACHE_DIR: {CACHE_DIR}")



# In[ ]:


def voxel_select(voxels: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Combine repeated voxel responses into a single vector.

    Args:
        voxels: Tensor of shape [B, V] (already selected) or [B, R, V] (R repeats)
        mode: "mean" to average repeats (default and only mode used in this notebook)

    Returns:
        Tensor of shape [B, V]
    """
    if voxels.ndim == 2:
        return voxels  # Already [B, V]

    if voxels.ndim != 3:
        raise ValueError(f"voxels must have shape [B,V] or [B,R,V], got {tuple(voxels.shape)}")

    # Average across repeats (dim=1)
    return voxels.mean(dim=1)


# In[ ]:


# --- Helper to build the streaming dataset ---
def build_nsd_dataset(
    subject_id: int,
    split: str,
    batch_size: int,
    seed: int = 42  # Add seed parameter for deterministic shuffle
):
    """
    Creates a streaming WebDataset pipeline.
    split: 'train', 'val', or 'test'
    seed: Random seed for deterministic shuffling (train split only)
    """
    # Use 'resolve' for direct file access
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_new"

    if split == "train":
        # pattern: train/train_subj0X_{0..17}.tar
        url_pattern = f"{base_url}/train/train_subj0{subject_id}_{{0..17}}.tar"
    elif split == "val":
        # pattern: val/val_subj0X_0.tar
        url_pattern = f"{base_url}/val/val_subj0{subject_id}_0.tar"
    elif split == "test":
        # pattern: test/test_subj0X_{0..1}.tar
        url_pattern = f"{base_url}/test/test_subj0{subject_id}_{{0..1}}.tar"
    else:
        raise ValueError(f"Unknown split: {split}")

    urls = list(braceexpand.braceexpand(url_pattern))

    # Use shardshuffle=False to ensure deterministic shard ordering
    dataset = wds.WebDataset(urls, resampled=False, shardshuffle=False)

    if split == "train":
        # Shuffle training data with deterministic random generator
        # This makes the shuffle reproducible across runs
        rng = random.Random(seed)
        dataset = dataset.shuffle(100, rng=rng)

    dataset = (
        dataset
        .decode("torch")
        .rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy")
        .to_tuple("voxels", "images", "trial")
        .batched(batch_size, partial=(split != "train"))
    )

    return dataset

# --- Main Loading Function ---
def get_dataloaders(cfg: LowLevelCfg):
    print(f"Setting up streaming dataloaders for Subject {cfg.subject_id}...")

    # 1. Train Loader
    # Train has ~18 shards, so multiple workers work well.
    # Pass seed for deterministic shuffling
    train_ds = build_nsd_dataset(cfg.subject_id, "train", cfg.batch_size, seed=cfg.seed)
    train_loader = wds.WebLoader(
        train_ds,
        batch_size=None, # Batching is handled in the dataset pipeline
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 2. Validation Loader
    # Val has only 1 shard. Using num_workers > 1 causes "No samples" error in empty workers.
    # We use num_workers=0 to run in the main process.
    val_ds = build_nsd_dataset(cfg.subject_id, "val", cfg.batch_size, seed=cfg.seed)
    val_loader = wds.WebLoader(
        val_ds,
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )

    # 3. Test Loader
    # Test has 2 shards. To be safe and simple, we also use 0 workers.
    test_ds = build_nsd_dataset(cfg.subject_id, "test", cfg.batch_size, seed=cfg.seed)
    test_loader = wds.WebLoader(
        test_ds,
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )

    print(f"Loaders ready. Batch size: {cfg.batch_size}")
    return train_loader, val_loader, test_loader


# ## Materializing Data from Streams
# 
# While streaming is memory-efficient, model fitting typically needs tensors we can index and shuffle. This means we "materialize" the data—iterate through the stream once and stack everything into arrays.
# 
# ### Dataset Sizes
# 
# For Subject 1, we're working with approximately:
# - **Training**: ~8,640 unique images (each shown 3 times = 25,920 total presentations)
# - **Validation**: 300 images
# - **Test**: 1,000 images (held out until final evaluation)
# 
# ### Why Repeats Matter
# 
# Each image was shown to the subject three times across different scanning sessions. This gives us voxel tensors with shape `[N, 3, 15724]`—three separate brain responses per stimulus. How we handle these repeats is a key design decision that differs between our linear and nonlinear models.

# In[ ]:


# --- Quick Data Preview (Optional) ---
# This cell demonstrates the data format.


train_loader, val_loader, test_loader = get_dataloaders(ll)

print("Fetching one batch to preview data format...")
for voxels, images, trials in train_loader:
    print(f"Voxel Shape: {voxels.shape}")  # [B, 3, Num_Voxels] - 3 repeats
    print(f"Image Shape: {images.shape}")  # [B, C, H, W]

    plt.figure(figsize=(4, 4))
    plt.imshow(images[0].permute(1, 2, 0))
    plt.title(f"Trial: {trials[0].item()}")
    plt.axis("off")
    plt.show()
    break


del train_loader, val_loader, test_loader


# # Part 1: Data Loading and Preprocessing
# 
# Our goal here is straightforward: produce three clean tensors for each split that we can feed into our models.
# 
# ## The Data Structure
# 
# For each trial we have:
# - **X**: fMRI voxels (one vector per trial, ~15k dimensions)
# - **I**: The stimulus image the subject was viewing
# - **trial_id**: Lets us group repeated presentations of the same stimulus
# 
# ## Handling Noisy Data
# 
# fMRI is inherently noisy. A single voxel measurement contains not just neural signal but also scanner drift, physiological noise (heartbeat, breathing), and random thermal fluctuations. Repeated presentations of the same stimulus are invaluable because they let us separate signal from noise.
# 
# We'll use repeats in two different ways depending on the model:
# 1. **Average repeats** for the linear baseline (reduces noise, cleaner targets)
# 2. **Expand repeats** for the nonlinear model (treat each repeat as a separate sample, 3× more training data)

# In[ ]:


import torch

def to_vae_range(images: torch.Tensor) -> torch.Tensor:
    """
    images: [B,3,H,W], uint8 or float
    returns float32 in [-1,1]
    """
    x = images
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    x = x * 2 - 1
    return x


# ## Materializing the Dataset
# 
# We stream through the WebDataset once and stack everything into tensors. This gives us simple, indexable arrays that work with standard PyTorch training loops.
# 
# ### Memory Optimization
# 
# Images are converted to `uint8` immediately after loading. This saves 4× memory compared to float32 (1 byte vs 4 bytes per pixel), which matters when you're loading thousands of 256×256 RGB images.
# 
# ### Deterministic Splits
# 
# By materializing and shuffling with a fixed seed, we ensure the exact same train/val/test split across runs. This is crucial for reproducibility—you want to know that improvements come from your model, not from lucky data shuffling.

# In[ ]:


import numpy as np
from tqdm.auto import tqdm

def take_n_samples(loader, n_samples=None, seed=42):
    """
    Takes n_samples from a streaming WebLoader.
    If n_samples is None, takes all available samples.
    Returns:
      X: [N,V], images: [N,3,H,W] (uint8), trials: [N]
    """
    set_seed(seed)

    xs, ims, trs = [], [], []
    n = 0

    # Add a progress bar. If n_samples is None, total is unknown.
    pbar = tqdm(total=n_samples, desc="Materializing samples", unit="samples")

    for vox, img, trial in loader:
        # Optimize memory: Convert images to uint8 immediately (saves 4x RAM)
        # img comes in as float [0,1]. Scale to [0,255] and cast.
        if img.dtype == torch.float32:
            img = (img * 255).clamp(0, 255).to(torch.uint8)

        b = vox.shape[0]
        xs.append(vox.cpu()) # Keep on CPU to save GPU memory
        ims.append(img.cpu())
        trs.append(trial.cpu())
        n += b
        pbar.update(b)

        if n_samples is not None and n >= n_samples:
            break

    pbar.close()

    if not xs:
        raise ValueError("Loader yielded no data!")

    X = torch.cat(xs, dim=0)
    I = torch.cat(ims, dim=0)
    T = torch.cat(trs, dim=0)

    if n_samples is not None:
        X = X[:n_samples]
        I = I[:n_samples]
        T = T[:n_samples]

    # Shuffle in memory
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    I = I[perm]
    T = T[perm]

    return X, I, T


# In[ ]:


# 1. Refresh global dataloaders with the updated config (ll)
train_loader, val_loader, test_loader = get_dataloaders(ll)

# 2. Materialize Data (Using optimized uint8 storage)
# We pass n_samples=None to consume the entire dataset defined by the loaders
print("Materializing Training Data...")
Xtr, Itr, _ = take_n_samples(train_loader, n_samples=None, seed=ll.seed)
print("Materializing Validation Data...")
Xva, Iva, _ = take_n_samples(val_loader,   n_samples=None,   seed=ll.seed + 1)
print("Materializing Test Data...")
Xte, Ite, Tte = take_n_samples(test_loader,  n_samples=None,  seed=ll.seed + 2)

print("Xtr:", Xtr.shape, "Itr:", Itr.shape)
print("Xva:", Xva.shape, "Iva:", Iva.shape)
print("Xte:", Xte.shape, "Ite:", Ite.shape)


# In[ ]:


# --- Prepare TWO versions of voxel data ---
# AVERAGED: For Ridge regression (reduces noise, better linear fit)
# EXPANDED: For MLPs (use all repeats as separate samples = 3x more data!)

print(f"Original shape: {Xtr.shape}  (N samples × 3 repeats × V voxels)")
N_tr, R, V = Xtr.shape

# --- 1. AVERAGED version for Ridge ---
print("\nCreating AVERAGED data for Ridge...")
Xtr_avg = voxel_select(Xtr, mode="mean")  # [N, V]
Xva_avg = voxel_select(Xva, mode="mean")
Xte_avg = voxel_select(Xte, mode="mean")
print(f"   Xtr_avg: {Xtr_avg.shape}")

# --- 2. EXPANDED version for MLPs (3x more samples!) ---
# Each repeat sees the same image, so we only expand voxels here
# The latents (Ztr_exp) will be replicated later - no extra VAE encoding needed!
print("\nCreating EXPANDED voxels for MLPs (3x samples)...")
Xtr_exp_raw = Xtr.view(N_tr * R, V)  # [N*3, V]
print(f"   Xtr_exp: {Xtr_exp_raw.shape} ({N_tr} samples × {R} repeats = {N_tr * R} total)")

# Validation/test: use averaged (cleaner evaluation)
Xva = Xva_avg
Xte = Xte_avg

# Ridge uses averaged
Xtr = Xtr_avg

def zscore_train_apply(Xtr, Xva, Xte, eps=1e-6):
    # Calculate stats on training set only
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, keepdim=True).clamp_min(eps)
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd, mu, sd

# Apply Z-score to averaged data (for Ridge)
Xtr, Xva, Xte, Xmu, Xsd = zscore_train_apply(Xtr.float(), Xva.float(), Xte.float())

# Apply Z-score to expanded data (for MLPs) using SAME stats
Xtr_exp = (Xtr_exp_raw.float() - Xmu) / Xsd

print(f"\nRidge will use: Xtr {Xtr.shape} (averaged)")
print(f"MLPs will use:  Xtr_exp {Xtr_exp.shape} (3x samples!)")


# ## Design Choice: Averaged vs Expanded Voxels
# 
# This is one of the most important decisions in the pipeline. We have 3 repeated measurements per image—how should we use them?
# 
# **For Ridge (linear model)**: We *average* the repeats. Ridge regression is sensitive to noise, and averaging reduces measurement noise by approximately $\sqrt{3}$. Cleaner inputs lead to more stable weight estimates.
# 
# **For MLP (nonlinear model)**: We *expand* the repeats into separate samples. Neural networks are data-hungry and have built-in regularization (dropout, weight decay) to handle noise. More samples—even noisy ones—often help more than cleaner but fewer samples.
# 
# This creates an interesting asymmetry: Ridge trains on ~8,640 samples, while MLP trains on ~25,920. The MLP sees noisier data but more of it.

# # Part 2: Image Targets via VAE Latents
# 
# As we discussed in Notebook 2, predicting raw pixels from fMRI is impractical due to the massive dimensionality mismatch and the harshness of pixel-space losses. Instead, we predict VAE latents: 4,096 dimensions instead of 196,608, with the added benefit that nearby latents correspond to visually similar images.
# 
# 
# $$
# \text{196,608 pixels} \xrightarrow{\text{VAE}} \text{4,096 latents}
# $$
# 
# 
# Here we put that into practice. We'll encode all NSD stimulus images into latent space using the same Stable Diffusion VAE from Notebook 2, and these latents become the regression targets that our brain decoders learn to predict from voxels.

# In[ ]:


import torch.nn.functional as F
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

@torch.inference_mode()
def encode_latents_sdvae(images: torch.Tensor, vae: AutoencoderKL, img_size: int, batch_size: int = 16):
    """
    returns scaled SD latents: [N, 4, img_size/8, img_size/8]
    """
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    out = []
    for i in tqdm(range(0, len(images), batch_size), desc="VAE encode"):
        x = images[i:i+batch_size]
        x = to_vae_range(x).to(device)

        if x.shape[-2:] != (img_size, img_size):
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

        dist = vae.encode(x).latent_dist
        z = dist.mode() * sf  # deterministic target in SD latent convention
        out.append(z.cpu())

    return torch.cat(out, dim=0)

@torch.inference_mode()
def decode_latents_sdvae(latents: torch.Tensor, vae: AutoencoderKL):
    """
    latents are expected scaled (SD convention). returns images in [0,1]
    """
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    z = (latents.to(device) / sf)
    x = vae.decode(z).sample
    x = (x.clamp(-1, 1) + 1) / 2
    return x.cpu()


# Encoding the latents might take a few minutes for a dataset of this size.

# In[ ]:


# Encode fresh latents to ensure they match current data order

print("Encoding latents from current images...")
vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()

# Encode averaged images (for Ridge)
print("  Encoding averaged images for Ridge...")
Ztr = encode_latents_sdvae(Itr, vae, ll.img_size, batch_size=16)
Zva = encode_latents_sdvae(Iva, vae, ll.img_size, batch_size=16)
Zte = encode_latents_sdvae(Ite, vae, ll.img_size, batch_size=16)

# Expand latents for MLPs by REPLICATING (no extra VAE encoding!)
# Each of the 3 voxel repeats saw the SAME image, so they share the same latent target
print("  Replicating latents for MLPs (no extra encoding!)...")
Ztr_exp = Ztr.unsqueeze(1).expand(-1, R, -1, -1, -1).reshape(N_tr * R, *Ztr.shape[1:])

vae.to("cpu")
torch.cuda.empty_cache()

print(f"\nZtr (Ridge): {Ztr.shape}")
print(f"Ztr_exp (MLPs): {Ztr_exp.shape}")
print(f"   Zva: {Zva.shape}, Zte: {Zte.shape}")


# ## Target Normalization
# 
# We z-score the latent targets for the same reason we z-scored the voxels earlier. Dimensions with larger scales would dominate the loss, even if smaller-scale dimensions carry equally important information. Normalizing ensures every latent channel contributes proportionally during training.

# In[ ]:


# Target normalization

def zscore_targets_train_apply(Ytr: torch.Tensor, Yva: torch.Tensor, Yte: torch.Tensor, eps: float = 1e-6):
    """Z-score targets using train stats only (per latent dimension)."""
    mu = Ytr.mean(dim=0, keepdim=True)
    sd = Ytr.std(dim=0, keepdim=True).clamp_min(eps)
    return (Ytr - mu) / sd, (Yva - mu) / sd, (Yte - mu) / sd, mu, sd

def unnormalize_targets(Y: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return Y * sd + mu


# # Part 3: Linear Baseline (Ridge Regression)
# 
# Ridge regression is the natural first baseline for brain decoding. fMRI data is high-dimensional (15k voxels), noisy, and highly correlated—exactly the conditions where unregularized linear regression falls apart.
# 
# ## Why Ridge?
# 
# Without regularization, linear regression tries to find weights that perfectly fit the training data. With more voxels than samples, there are infinitely many solutions that achieve zero training error—most of which generalize terribly.
# 
# Ridge adds an L2 penalty that shrinks weights toward zero:
# 
# $$\arg\min_W \|XW - Z\|_2^2 + \alpha\|W\|_2^2$$
# 
# The hyperparameter $\alpha$ controls how much we trust the data versus how much we prefer small weights. Higher $\alpha$ means more shrinkage, more stable predictions, but potentially underfitting.
# 
# ## A Note on Expectations
# 
# If your MLP doesn't beat Ridge, that's not failure. It often means either the relationship is mostly linear, or the MLP is overfitting. Ridge is a strong baseline precisely because it's stable and interpretable. Although in this case spesifically we would expect high-variance methods to do better in general.

# In[ ]:


class DualRidge:
    def __init__(self, alpha: float = 1e5):
        self.alpha = alpha
        self.Xtr = None
        self.A = None  # [N,D]

    def fit(self, Xtr: torch.Tensor, Ytr: torch.Tensor):
        """
        Xtr: [N,V]
        Ytr: [N,D]
        """
        Xtr = Xtr.float()
        Ytr = Ytr.float()
        N = Xtr.shape[0]

        K = Xtr @ Xtr.T                       # [N,N]
        K = K + self.alpha * torch.eye(N)     # ridge
        A = torch.linalg.solve(K, Ytr)        # [N,D]

        self.Xtr = Xtr
        self.A = A
        return self

    def predict(self, X: torch.Tensor):
        """
        X: [M,V]
        returns: [M,D]
        """
        X = X.float()
        Kxt = X @ self.Xtr.T                  # [M,N]
        return Kxt @ self.A                   # [M,D]


# ## The Dual Formulation
# 
# We use the "dual" form of ridge regression, which works with an $N \times N$ matrix instead of a $V \times V$ matrix, where $N$ is the number of samples and $V$ is the number of voxels. Since we have many more voxels (15k) than samples (8k), the dual formulation is a little faster to compute. The math isn't critical for this tutorial.

# In[ ]:


# Check for shape mismatch between voxels (Xtr) and latents (Ztr)
# This happens if n_train is changed but Ztr is loaded from a stale cache.
if Xtr.shape[0] != Ztr.shape[0]:
    print(f"Shape mismatch detected! Xtr: {Xtr.shape[0]}, Ztr: {Ztr.shape[0]}.")
    print("Re-encoding training latents to match new data size...")

    # Re-load VAE and encode
    vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()
    Ztr = encode_latents_sdvae(Itr, vae, ll.img_size, batch_size=16)

    # Cleanup
    vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"New Ztr shape: {Ztr.shape}")

# --- Flatten latents (targets) ---
Ytr = Ztr.flatten(1)  # [N, 4*32*32] when img_size=256
Yva = Zva.flatten(1)
Yte = Zte.flatten(1)

# --- Normalize targets (improves conditioning, often boosts SSIM) ---
Ytr_n, Yva_n, _, Ymu, Ysd = zscore_targets_train_apply(Ytr.float(), Yva.float(), Yte.float())
print("Target normalization:", "Ymu", tuple(Ymu.shape), "Ysd", tuple(Ysd.shape))


# ## Hyperparameter Tuning: Ridge Alpha
# 
# Before fitting our final model, we need to choose the regularization strength $\alpha$. This is Ridge's only hyperparameter, but it matters a lot—too low and the model overfits; too high and it underfits.
# 
# We do a simple grid search on the validation set. This is cheap for Ridge (each fit should take about 20-30 seconds) and gives us confidence we're using a reasonable value.

# In[ ]:


# --- Alpha Grid Search (on normalized latent targets) ---
alphas = [10000, 50000, 80000, 100000]
best_alpha = None
best_mse = float("inf")

print(f"Searching for best alpha among: {alphas}...")

for a in alphas:
    # Fit Ridge on normalized targets
    ridge_tmp = DualRidge(alpha=a).fit(Xtr, Ytr_n)
    # Predict on Validation (normalized)
    Yva_hat_tmp_n = ridge_tmp.predict(Xva)
    # MSE in normalized space (comparable across dims)
    mse = torch.mean((Yva_hat_tmp_n - Yva_n)**2).item()
    print(f"Alpha: {a:.0f} | Val MSE (norm): {mse:.6f}")
    if mse < best_mse:
        best_mse = mse
        best_alpha = a

print(f"\nBest Alpha: {best_alpha:.0f} (MSE norm: {best_mse:.6f})")

# --- Fit final Ridge with best alpha ---
ridge = DualRidge(alpha=best_alpha).fit(Xtr, Ytr_n)
Yva_hat_n = ridge.predict(Xva)
print("Pred val latent flat (normalized):", Yva_hat_n.shape)

# --- Unnormalize for reshaping / decoding / metrics ---
Yva_hat = unnormalize_targets(Yva_hat_n, Ymu, Ysd)
Zva_hat = Yva_hat.view_as(Zva)


# In[ ]:


print("GT latents:  mean", Zva.mean().item(), "std", Zva.std().item())
print("PRED latents: mean", Zva_hat.mean().item(), "std", Zva_hat.std().item())

# Also compare per-sample norm
gt_n = Zva.flatten(1).norm(dim=1).mean().item()
pr_n = Zva_hat.flatten(1).norm(dim=1).mean().item()
print("Mean ||z||  GT:", gt_n, "PRED:", pr_n)


# ## First Results: Ridge Reconstructions
# 
# Let's see what our linear baseline produces. The left column shows ground truth (what the VAE itself can reconstruct), and the right shows our predictions from brain activity.
# 
# Don't expect photorealism—we're predicting from ~15k noisy voxels. Look for whether the coarse structure (colors, layout, major shapes) is captured.

# ## Evaluation Metrics
# 
# We report multiple metrics because each one can be misleading in different ways.
# 
# **Latent MSE** tells us how well we matched the target representation—useful for debugging training, but doesn't directly measure image quality.
# 
# **Pixel MSE** is strict per-pixel error after decoding. It penalizes small spatial shifts harshly, even when the image "looks" correct.
# 
# **PSNR** (Peak Signal-to-Noise Ratio) is just MSE expressed in decibels. It's easier to compare across different settings but has the same limitations.
# 
# **SSIM** (Structural Similarity Index) measures local structure: luminance, contrast, and structural patterns. It often correlates better with human perception than MSE. Two images can have identical MSE but very different SSIM if one preserves structure while the other is uniformly blurry.
# 
# The "best" metric depends on your goals. For this tutorial, we focus on SSIM as the primary quality measure.

# In[ ]:


from pytorch_msssim import ssim as compute_ssim
import gc

def resize01(images, size):
    x = images.float()
    if x.max() > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x

# --- Memory Cleanup ---
gc.collect()
torch.cuda.empty_cache()

# Load VAE once for both visualization and evaluation
vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()

# --- 1. VISUALIZATION: First 8 samples ---
print("Decoding first 8 samples for visualization...")
coarse = decode_latents_sdvae(Zva_hat[:8], vae)
gt_viz = decode_latents_sdvae(Zva[:8], vae)

fig, axes = plt.subplots(8, 2, figsize=(6, 18))
for i in range(8):
    axes[i,0].imshow(gt_viz[i].permute(1,2,0).clamp(0,1))
    axes[i,0].set_title("GT VAE decode")
    axes[i,0].axis("off")
    axes[i,1].imshow(coarse[i].permute(1,2,0).clamp(0,1))
    axes[i,1].set_title("Pred coarse")
    axes[i,1].axis("off")
plt.tight_layout()
plt.show()
del coarse, gt_viz
torch.cuda.empty_cache()

# --- 2. EVALUATION: Full validation set ---
print("\nComputing Ridge metrics...")

# Compute latent MSE
latent_mse = torch.mean((Zva_hat - Zva) ** 2).item()
print(f"Ridge Latent MSE: {latent_mse:.6f}")

# Prepare ground truth images (keep for MLP evaluation later)
gt_imgs = resize01(Iva, ll.img_size)
del Iva
gc.collect()

# Compute metrics in batches
ssim_scores = []
mse_scores = []
BATCH_SIZE = 10

for batch_start in tqdm(range(0, len(Zva_hat), BATCH_SIZE), desc="Evaluating Ridge"):
    batch_end = min(batch_start + BATCH_SIZE, len(Zva_hat))

    pred_batch = decode_latents_sdvae(Zva_hat[batch_start:batch_end], vae).cpu()
    gt_batch = gt_imgs[batch_start:batch_end]

    for i in range(len(pred_batch)):
        ssim_val = compute_ssim(pred_batch[i:i+1], gt_batch[i:i+1], data_range=1.0, size_average=True).item()
        mse_val = torch.mean((pred_batch[i:i+1] - gt_batch[i:i+1]) ** 2).item()
        ssim_scores.append(ssim_val)
        mse_scores.append(mse_val)

    del pred_batch, gt_batch
    torch.cuda.empty_cache()

pixel_ssim = float(np.mean(ssim_scores))
pixel_mse = float(np.mean(mse_scores))

# Cleanup VAE (keep gt_imgs for MLP)
vae.to("cpu")
del vae
gc.collect()
torch.cuda.empty_cache()

print(f"\nRidge Results:")
print(f"  Pixel  MSE: {pixel_mse:.6f}")
print(f"  SSIM      : {pixel_ssim:.4f}")


# # Part 4: Nonlinear Decoder (MLP)
# 
# The mapping from voxels to image features might not be purely linear. An MLP lets us capture potential nonlinear interactions between voxels that Ridge misses.
# 
# ## Key Differences from Ridge
# 
# The MLP differs from Ridge in three important ways. First, we expand the training data by treating each of the three voxel repeats as a separate sample, giving us roughly 25,920 training examples instead of 8,640. Neural networks benefit from this because they have the capacity to learn through the noise, whereas Ridge benefits more from cleaner averaged inputs.
# 
# Second, we replace the flat output with a CNN upsampler that respects the spatial structure of VAE latents. Ridge predicts each latent dimension independently, but the MLP backbone feeds into convolutional layers that enforce local spatial coherence in the predicted latent maps.
# 
# Third, we use aggressive regularization across multiple axes: high dropout, mixup augmentation, input noise, voxel masking, and weight decay. These work together to prevent the network from memorizing training examples while still allowing it to capture nonlinear relationships that Ridge cannot represent.

# In[ ]:


@dataclass
class MLPConfig:
    hidden_dims: Tuple[int, ...] = (1024, 2048, 2048)
    dropout: float = 0.4
    lr: float = 1.49e-4
    weight_decay: float = 3.73e-4
    batch_size: int = 128
    epochs: int = 100
    patience: int = 25
    noise_std: float = 0.06
    target_noise: float = 0
    mixup_alpha: float = 0.4
    grad_weight: float = 0.025
    mask_rate: float = 0.2

mlp_cfg = MLPConfig()
print(mlp_cfg)


# ## MLP Architecture Choices
# 
# Our MLP has two stages. The backbone is a standard feedforward network that maps the 15,724-dimensional voxel vector down through three hidden layers (1024, 2048, 2048) to a compact spatial representation. Each hidden layer uses LayerNorm for stable training across voxel dimensions, GELU activations, and dropout at 0.4. The backbone projects to a 64-channel 8x8 feature grid, which a small CNN upsampler then reshapes into the target 4-channel 32x32 latent map through two transposed convolutions.
# 
# The CNN upsampler matters because VAE latents have spatial structure. Adjacent positions in the 32x32 latent grid correspond to neighboring image regions. A flat MLP output would predict each of those 4,096 positions independently, ignoring the fact that nearby latent values should be correlated. The CNN head enforces local spatial coherence through its convolutional kernels, producing smoother latent maps that decode into cleaner images. The final output shape is identical to the flat case, so nothing downstream changes.

# In[ ]:


import torch.nn as nn

class VoxelMLP(nn.Module):
    """MLP backbone with CNN upsampler for spatially coherent VAE latent prediction."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int, ...], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 64 * 8 * 8))
        self.backbone = nn.Sequential(*layers)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 4, 3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).view(-1, 64, 8, 8)
        return self.upsample(h).flatten(1)


# ## Spatial Gradient Loss
# 
# Beyond simple MSE, we add a loss term that encourages the model to preserve spatial structure in the latent maps. This computes edges in both predicted and target latents using Sobel filters, then penalizes differences. The intuition is that two predictions might have the same MSE, but one preserves sharp boundaries while the other is blurry. The gradient loss prefers the sharp one, which often improves SSIM scores without requiring expensive image-space decoding during training.

# In[ ]:


import kornia
def latent_spatial_grad_loss(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    latent_shape: tuple[int, int, int] = (4, 32, 32),
    reduction: str = "mean",
    loss_type: str = "l1",
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Encourages matching *spatial structure* in latent maps (helps sharpness/SSIM)
    without decoding to pixel space. Very cheap vs VAE-decode losses.

    pred_flat/target_flat: [B, D] where D = C*H*W
    latent_shape: (C,H,W)
    """
    if weight <= 0:
        return pred_flat.new_tensor(0.0)
    C, H, W = latent_shape
    pred = pred_flat.view(-1, C, H, W)
    target = target_flat.view(-1, C, H, W)
    # spatial_gradient returns [B, C, 2, H, W] (dy, dx)
    gp = kornia.filters.spatial_gradient(pred, mode="sobel", order=1)
    gt = kornia.filters.spatial_gradient(target, mode="sobel", order=1)
    if loss_type == "l2":
        base = (gp - gt).pow(2)
        loss = base.mean() if reduction == "mean" else base.sum()
    else:
        base = (gp - gt).abs()
        loss = base.mean() if reduction == "mean" else base.sum()
    return loss * weight


# In[ ]:


from torch.utils.data import TensorDataset, DataLoader

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup augmentation: blend pairs of samples to create virtual training data."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1 - lam) * x[index]
    y_mixed = lam * y + (1 - lam) * y[index]
    return x_mixed, y_mixed

def train_mlp(
    model: nn.Module,
    Xtr: torch.Tensor, Ytr: torch.Tensor,
    Xva: torch.Tensor, Yva: torch.Tensor,
    cfg,
    device: torch.device,
    noise_std: float = 0.1,
    target_noise: float = 0.0,
    mixup_alpha: float = 0.0,
    grad_weight: float = 0.0,
    mask_rate: float = 0.0,
    latent_shape: Optional[Tuple[int, int, int]] = None,
):
    model = model.to(device)

    base_lr = float(getattr(cfg, 'lr', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=base_lr * 0.01)

    train_ds = TensorDataset(Xtr, Ytr)
    default_workers = 0 if os.name == "nt" else 2
    num_workers = int(getattr(cfg, "num_workers", default_workers))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) if num_workers > 0 else False,
    )

    Xva_dev = Xva.to(device)
    Yva_dev = Yva.to(device)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    pbar = tqdm(range(cfg.epochs), desc="Training MLP")

    for epoch in pbar:
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if mixup_alpha > 0:
                xb, yb = mixup_batch(xb, yb, alpha=mixup_alpha)
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
            if target_noise > 0:
                yb = yb + torch.randn_like(yb) * target_noise
            if mask_rate > 0:
                mask = (torch.rand_like(xb) > mask_rate).float()
                xb = xb * mask

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.huber_loss(pred, yb)
            if grad_weight > 0 and latent_shape is not None:
                loss = loss + latent_spatial_grad_loss(pred, yb, latent_shape=latent_shape, weight=grad_weight)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_dev)
            val_loss = F.huber_loss(pred_va, Yva_dev).item()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        gap = avg_train_loss - val_loss
        pbar.set_postfix({'train': f'{avg_train_loss:.4f}', 'val': f'{val_loss:.4f}', 'gap': f'{gap:.4f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    print(f"Best val loss: {best_val_loss:.6f}")
    return model, history


# ## Training Loop and Regularization
# 
# Neural networks on fMRI data love to overfit. We use multiple complementary strategies to fight this, and the combination matters more than any single technique.
# 
# We train with Huber loss rather than standard MSE. Huber loss behaves like MSE for small errors but switches to L1 for large errors, which prevents individual badly-predicted latent dimensions from dominating the gradient. In practice this produces predictions that are less conservative and more willing to commit to spatial structure, which improves SSIM.
# 
# For regularization, we layer several techniques that each address a different failure mode. Mixup blending at $\alpha = 0.4$ creates virtual training examples by interpolating pairs of voxel-latent samples, smoothing the learned mapping and reducing overfitting. Gaussian input noise at $\sigma = 0.06$ prevents the model from memorizing exact voxel values that may reflect scanner noise rather than neural signal. Voxel masking randomly zeros out 20% of input voxels each forward pass, forcing the network to build redundant representations across the voxel population rather than relying on any single voxel. This is conceptually similar to dropout but applied to the input features, and it complements the 0.4 dropout applied to hidden layers.
# 
# Early stopping monitors validation loss and restores the best checkpoint if no improvement is seen for 25 epochs. Cosine learning rate decay gradually reduces the learning rate to 1% of its initial value, allowing large early updates to find the general solution and small later updates to refine without overshooting.

# ## Training the MLP
# 
# Now we fit the MLP on the expanded training set and compare against Ridge.
# 
# When evaluating, keep in mind that improvements might be metric-specific. The MLP might improve SSIM but worsen pixel MSE, or vice versa. This usually reflects different error modes rather than a contradiction—preserving structure isn't the same as minimizing per-pixel error.

# In[ ]:


# Train Simple MLP with strong regularization
set_seed(ll.seed)

latent_shape = (int(Ztr.shape[1]), int(Ztr.shape[2]), int(Ztr.shape[3]))
print("Latent shape:", latent_shape)

# Use expanded data for MLP training (3x samples!)
Ytr_exp_mlp = Ztr_exp.flatten(1).float()
Ytr_exp_mlp_n = (Ytr_exp_mlp - Ymu) / Ysd
print("Training target space: normalized latents")



in_dim  = Xtr.shape[1]
out_dim = Ytr_n.shape[1]
model = VoxelMLP(in_dim, out_dim, mlp_cfg.hidden_dims, mlp_cfg.dropout)
print(f"Config: {mlp_cfg}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training with EXPANDED data: {Xtr_exp.shape[0]} samples")
print(f"Regularization: noise={mlp_cfg.noise_std}, target_noise={mlp_cfg.target_noise}, mixup={mlp_cfg.mixup_alpha}")

model, history = train_mlp(
    model,
    Xtr_exp, Ytr_exp_mlp_n,
    Xva, Yva_n,
    mlp_cfg, device,
    noise_std=mlp_cfg.noise_std,
    target_noise=getattr(mlp_cfg, 'target_noise', 0.0),
    mixup_alpha=getattr(mlp_cfg, 'mixup_alpha', 0.0),
    grad_weight=mlp_cfg.grad_weight,
    mask_rate=mlp_cfg.mask_rate,
    latent_shape=latent_shape,
)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('MLP Training (strong regularization)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['lr'], label='Learning Rate', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('LR')
ax2.set_title('Learning Rate (cosine decay)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:


# Generate MLP predictions (in normalized latent space)
model.eval()
with torch.no_grad():
    Yva_hat_mlp_n = model(Xva.to(device)).cpu()

# Unnormalize for decoding/metrics
Yva_hat_mlp = unnormalize_targets(Yva_hat_mlp_n, Ymu, Ysd)
Zva_hat_mlp = Yva_hat_mlp.view_as(Zva)

print("MLP predicted latents:")
print(f"  mean: {Zva_hat_mlp.mean().item():.4f}, std: {Zva_hat_mlp.std().item():.4f}")
print(f"  GT mean: {Zva.mean().item():.4f}, GT std: {Zva.std().item():.4f}")


# In[ ]:


# Evaluate MLP predictions (batched for speed)
print("Evaluating MLP predictions...")

# Compute latent MSE
latent_mse_mlp = torch.mean((Zva_hat_mlp - Zva) ** 2).item()

# Load VAE and compute metrics in batches
vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()

ssim_scores_mlp = []
mse_scores_mlp = []
BATCH_SIZE = 10

for batch_start in tqdm(range(0, len(Zva_hat_mlp), BATCH_SIZE), desc="Evaluating MLP"):
    batch_end = min(batch_start + BATCH_SIZE, len(Zva_hat_mlp))

    pred_batch = decode_latents_sdvae(Zva_hat_mlp[batch_start:batch_end], vae).cpu()
    gt_batch = gt_imgs[batch_start:batch_end]

    for i in range(len(pred_batch)):
        ssim_val = compute_ssim(pred_batch[i:i+1], gt_batch[i:i+1], data_range=1.0, size_average=True).item()
        mse_val = torch.mean((pred_batch[i:i+1] - gt_batch[i:i+1]) ** 2).item()
        ssim_scores_mlp.append(ssim_val)
        mse_scores_mlp.append(mse_val)

    del pred_batch, gt_batch
    torch.cuda.empty_cache()

pixel_ssim_mlp = float(np.mean(ssim_scores_mlp))
pixel_mse_mlp = float(np.mean(mse_scores_mlp))

# Cleanup
vae.to("cpu")
del vae, gt_imgs  # Now we can delete gt_imgs
gc.collect()
torch.cuda.empty_cache()

print("\nMLP Results:")
print(f"  Latent MSE: {latent_mse_mlp:.6f}")
print(f"  Pixel  MSE: {pixel_mse_mlp:.6f}")
print(f"  SSIM      : {pixel_ssim_mlp:.4f}")

print("\nRidge Results (for comparison):")
print(f"  Latent MSE: {latent_mse:.6f}")
print(f"  SSIM      : {pixel_ssim:.4f}")

improvement = (pixel_ssim_mlp - pixel_ssim) / max(pixel_ssim, 1e-8) * 100
print(f"\nSSIM change vs Ridge: {improvement:+.2f}%")


# ## Validation Summary: Ridge vs MLP
# 
# Both models have their strengths.
# 
# **Ridge** is fast, stable, and interpretable. It uses repeat-averaged data and has a single hyperparameter. When the underlying relationship is mostly linear, Ridge is hard to beat.
# 
# **MLP** can capture nonlinear interactions and uses 3× more training data (expanded repeats). But it requires careful regularization and early stopping to avoid memorizing the training set.
# 
# The validation comparison tells us which approach works better for this subject and dataset. But the real test comes next—evaluating on completely held-out data.

# In[ ]:


# --- Summary ---
print("=" * 60)
print("LOW-LEVEL BRAIN DECODING SUMMARY")
print("=" * 60)
print()
print("Approach: Decode VAE latents from fMRI voxels")
print(f"Subject: {ll.subject_id}")
print(f"Train samples: {len(Xtr)}")
print(f"Val samples: {len(Xva)}")
print()
print("MODELS:")
print(f"  1. Ridge Regression (alpha={ll.ridge_alpha:.0f})")
print(f"  2. MLP (neural network)")
print()
print("VALIDATION RESULTS:")
print(f"  Ridge:  SSIM = {pixel_ssim:.4f}")
print(f"  MLP:    SSIM = {pixel_ssim_mlp:.4f}")
print()
improvement = (pixel_ssim_mlp - pixel_ssim) / max(pixel_ssim, 1e-8) * 100
print(f"MLP improvement over Ridge: {improvement:+.2f}%")
print("=" * 60)


# # Part 5: Final Test Set Evaluation
# 
# The test set contains images never used during training or model selection. This is the only place where we should report "final" metrics—everything before this was tuning and comparison.
# 
# We'll decode both Ridge and MLP predictions back to images, compute metrics across all test samples, and visualize reconstructions for qualitative assessment. Numbers tell part of the story; seeing actual images tells the rest.

# In[ ]:


from pytorch_msssim import ssim as compute_ssim
import torch.nn.functional as F
import gc

# --- Memory Cleanup ---
gc.collect()
torch.cuda.empty_cache()

print("Evaluating on TEST SET")
print(f"   Test samples: {len(Xte)}")

# --- Generate latent predictions on TEST set ---
print("\nGenerating predictions...")
Yte_ridge = ridge.predict(Xte)
Yte_ridge_unnorm = unnormalize_targets(torch.tensor(Yte_ridge).float(), Ymu, Ysd)
Zte_hat_ridge = Yte_ridge_unnorm.view(-1, *Zte.shape[1:])
del Yte_ridge, Yte_ridge_unnorm
gc.collect()

model.eval()
with torch.no_grad():
    Yte_hat_mlp_n = model(Xte.to(device)).cpu()
Yte_hat_mlp = unnormalize_targets(Yte_hat_mlp_n, Ymu, Ysd)
Zte_hat_mlp = Yte_hat_mlp.view(-1, *Zte.shape[1:])
del Yte_hat_mlp_n, Yte_hat_mlp
gc.collect()

# --- Compute metrics in small batches (memory-efficient) ---
print("\nComputing metrics in batches of 10 (memory-efficient)...")

vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()

# Store metrics for each sample
all_metrics = []
batch_size = 10

for batch_start in tqdm(range(0, len(Zte), batch_size), desc="Test evaluation"):
    batch_end = min(batch_start + batch_size, len(Zte))

    # Decode batch for Ridge, MLP, and GT
    pred_ridge_batch = decode_latents_sdvae(Zte_hat_ridge[batch_start:batch_end], vae).cpu()
    pred_mlp_batch = decode_latents_sdvae(Zte_hat_mlp[batch_start:batch_end], vae).cpu()
    gt_batch = decode_latents_sdvae(Zte[batch_start:batch_end], vae).cpu()

    # Compute metrics for each sample in batch
    for i in range(len(gt_batch)):
        idx = batch_start + i
        pred_ridge = pred_ridge_batch[i:i+1]
        pred_mlp = pred_mlp_batch[i:i+1]
        gt = gt_batch[i:i+1]

        ridge_ssim = compute_ssim(pred_ridge, gt, data_range=1.0, size_average=True).item()
        ridge_mse = F.mse_loss(pred_ridge, gt).item()
        ridge_psnr = 10 * np.log10(1.0 / max(ridge_mse, 1e-10))

        mlp_ssim = compute_ssim(pred_mlp, gt, data_range=1.0, size_average=True).item()
        mlp_mse = F.mse_loss(pred_mlp, gt).item()
        mlp_psnr = 10 * np.log10(1.0 / max(mlp_mse, 1e-10))

        all_metrics.append({
            'idx': idx,
            'ridge_ssim': ridge_ssim, 'ridge_psnr': ridge_psnr, 'ridge_mse': ridge_mse,
            'mlp_ssim': mlp_ssim, 'mlp_psnr': mlp_psnr, 'mlp_mse': mlp_mse
        })

    # Cleanup batch
    del pred_ridge_batch, pred_mlp_batch, gt_batch
    torch.cuda.empty_cache()
    gc.collect()

# Cleanup VAE
vae.to("cpu")
del vae
gc.collect()
torch.cuda.empty_cache()

# --- Aggregate and Print Summary Metrics ---
ridge_ssims = np.array([m['ridge_ssim'] for m in all_metrics])
mlp_ssims = np.array([m['mlp_ssim'] for m in all_metrics])
ridge_psnrs = np.array([m['ridge_psnr'] for m in all_metrics])
mlp_psnrs = np.array([m['mlp_psnr'] for m in all_metrics])
ridge_mses = np.array([m['ridge_mse'] for m in all_metrics])
mlp_mses = np.array([m['mlp_mse'] for m in all_metrics])

print("\n" + "=" * 70)
print(f"TEST SET METRICS (All {len(all_metrics)} Samples)")
print("=" * 70)
print(f"{'Metric':<12} {'Ridge':<25} {'Simple MLP':<25}")
print("-" * 70)
print(f"{'SSIM':<12} {ridge_ssims.mean():.4f} +/- {ridge_ssims.std():.4f}        {mlp_ssims.mean():.4f} +/- {mlp_ssims.std():.4f}")
print(f"{'PSNR':<12} {ridge_psnrs.mean():.2f} +/- {ridge_psnrs.std():.2f} dB       {mlp_psnrs.mean():.2f} +/- {mlp_psnrs.std():.2f} dB")
print(f"{'MSE':<12} {ridge_mses.mean():.5f} +/- {ridge_mses.std():.5f}    {mlp_mses.mean():.5f} +/- {mlp_mses.std():.5f}")
print("=" * 70)

# Compute improvement
ssim_diff = mlp_ssims.mean() - ridge_ssims.mean()
ssim_pct = ssim_diff / ridge_ssims.mean() * 100
print(f"\nMLP vs Ridge: SSIM {ssim_diff:+.4f} ({ssim_pct:+.2f}%)")
print(f"   MLP wins on {np.sum(mlp_ssims > ridge_ssims)}/{len(mlp_ssims)} samples ({np.sum(mlp_ssims > ridge_ssims)/len(mlp_ssims)*100:.1f}%)")


# ## Visual Inspection
# 
# Metrics are useful but incomplete. Reconstructions can fail in ways that metrics don't capture, and they can succeed in ways that metrics undervalue.
# 
# When examining reconstructions, look for:
# - Do edges roughly align with the original?
# - Are colors plausible (not inverted or shifted)?
# - Does the reconstruction show meaningful structure, or has it collapsed to an average-looking blob?
# 
# When comparing Ridge and MLP, look for *systematic* differences rather than cherry-picking individual examples where one method happened to do better.

# In[ ]:


# Select 50 random samples from TEST set
np.random.seed(42)
n_samples = min(50, len(all_metrics))
random_indices = np.random.choice(len(all_metrics), size=n_samples, replace=False)
random_indices = np.sort(random_indices)

# Get metrics for selected samples
metrics_50 = [all_metrics[idx] for idx in random_indices]

# Decode selected samples for visualization
print(f"Decoding {n_samples} samples for visualization...")
vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()

viz_samples = []
for idx in tqdm(random_indices, desc="Decoding viz samples"):
    ridge_img = decode_latents_sdvae(Zte_hat_ridge[idx:idx+1], vae).cpu()
    mlp_img = decode_latents_sdvae(Zte_hat_mlp[idx:idx+1], vae).cpu()
    gt_img = decode_latents_sdvae(Zte[idx:idx+1], vae).cpu()
    viz_samples.append((gt_img[0], ridge_img[0], mlp_img[0]))

vae.to("cpu")
del vae
gc.collect()
torch.cuda.empty_cache()

samples_per_row = 5
n_rows = (n_samples + samples_per_row - 1) // samples_per_row

fig, axes = plt.subplots(n_rows, samples_per_row * 3, figsize=(samples_per_row * 6, n_rows * 2.2))

for i, (gt, ridge, mlp) in enumerate(viz_samples):
    row = i // samples_per_row
    col_base = (i % samples_per_row) * 3

    m = metrics_50[i]
    idx = random_indices[i]

    # Ground Truth
    ax_gt = axes[row, col_base]
    ax_gt.imshow(gt.permute(1, 2, 0).clamp(0, 1))
    ax_gt.set_title(f"GT #{idx}", fontsize=8)
    ax_gt.axis('off')

    # Ridge
    ax_ridge = axes[row, col_base + 1]
    ax_ridge.imshow(ridge.permute(1, 2, 0).clamp(0, 1))
    ax_ridge.set_title(f"Ridge\n{m['ridge_ssim']:.3f}", fontsize=7)
    ax_ridge.axis('off')

    # MLP
    ax_mlp = axes[row, col_base + 2]
    ax_mlp.imshow(mlp.permute(1, 2, 0).clamp(0, 1))
    ax_mlp.set_title(f"MLP\n{m['mlp_ssim']:.3f}", fontsize=7)
    ax_mlp.axis('off')

# Hide any unused axes
for i in range(len(viz_samples), n_rows * samples_per_row):
    row = i // samples_per_row
    col_base = (i % samples_per_row) * 3
    for j in range(3):
        axes[row, col_base + j].axis('off')

plt.suptitle("TEST SET: 50 Random Samples - Ground Truth | Ridge (SSIM) | MLP (SSIM)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

print(f"Displayed {n_samples} randomly selected TEST reconstructions")


# # Part 6: Saving Low-Level Reconstructions
# 
# We need to save our low-level reconstructions for use in the high-level pipeline. The next notebook will use these blurry reconstructions as a starting point for diffusion-based refinement (img2img).
# 
# We'll generate predictions for the **test set** using our best model (MLP), then save them in a format that's easy to load later. The downstream notebooks (Notebook 5) only need test set reconstructions.
# 
# ## Storage Options
# 
# **Google Drive (Recommended)**: If you're running on Colab, mounting your Drive makes it easy to persist outputs across sessions and share between notebooks.
# 
# **Local Storage**: If you're running locally or prefer not to use Drive, we'll save to a local directory that you can download manually.

# ## Memory Cleanup Before Part 6
# 
# Before we start saving, let's clean up large temporary variables we no longer need to free up RAM. The expanded training data and intermediate predictions can be deleted.

# In[ ]:


import gc

print("Cleaning up memory before Part 6...\n")

# Variables to delete - expanded data and intermediate predictions
# We keep: model, Zte_hat_mlp (for saving), Xmu/Xsd/Ymu/Ysd (normalization)
vars_to_delete = [
    # Expanded training data (large!)
    'Xtr_exp', 'Ztr_exp', 'Ytr_exp_mlp_n',
    # Validation predictions (already evaluated)
    'Zva_hat', 'Zva_hat_mlp',
    # Ridge test predictions (we only save MLP)
    'Zte_hat_ridge',
    # Training voxels (not needed for saving test predictions)
    'Xtr',
    # Original images (should already be encoded to latents)
    'Itr', 'Ite',
    # Ground truth images (used during evaluation)
    'gt_imgs', 'Iva',
]

print("Deleting intermediate artifacts:")
total_freed = 0
for var_name in vars_to_delete:
    if var_name in globals():
        var = globals()[var_name]
        if hasattr(var, 'numel') and hasattr(var, 'element_size'):
            size_mb = var.numel() * var.element_size() / 1e6
            print(f"  {var_name}: {var.shape} = {size_mb:.1f} MB")
            total_freed += size_mb
        del globals()[var_name]

gc.collect()
torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"Total freed: ~{total_freed:.0f} MB")
print(f"{'='*60}")

# Show what we're keeping for saving
print("\nKept for Part 6:")
kept_vars = ['model', 'Zte_hat_mlp', 'Zte', 'Xmu', 'Xsd', 'Ymu', 'Ysd', 'Tte']
for var_name in kept_vars:
    if var_name in globals():
        var = globals()[var_name]
        if hasattr(var, 'numel'):
            size_mb = var.numel() * var.element_size() / 1e6
            print(f"  {var_name}: {size_mb:.1f} MB")
        else:
            print(f"  {var_name}: (model)")

print("\nMemory cleanup complete! ✓")


# In[ ]:


USE_DRIVE = True

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_DIR = Path('/content/drive/MyDrive/NSD_Reconstructions')
        print("Google Drive mounted successfully!")
    except ImportError:
        print("Not running on Colab. Falling back to local storage.")
        USE_DRIVE = False
        SAVE_DIR = RESULTS_DIR / "reconstructions"
else:
    SAVE_DIR = RESULTS_DIR / "reconstructions"

# Create output directories
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LATENTS_DIR = SAVE_DIR / f"subj{ll.subject_id:02d}" / "latents"
IMAGES_DIR = SAVE_DIR / f"subj{ll.subject_id:02d}" / "images"
LATENTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Saving outputs to: {SAVE_DIR}")
print(f"  Latents: {LATENTS_DIR}")
print(f"  Images:  {IMAGES_DIR}")


# ## Reusing Test Set Predictions
# 
# We already generated MLP predictions on the test set in Part 5 (`Zte_hat_mlp`). We'll reuse those predictions for decoding—no need to recompute.

# In[ ]:


# Reuse predictions from Part 5 (already computed as Zte_hat_mlp)
print("Reusing MLP test predictions from Part 5...")

if 'Zte_hat_mlp' in dir():
    Zte_hat_final = Zte_hat_mlp
    print(f"  Test latents: {Zte_hat_final.shape}")
else:
    # Fallback: regenerate if Part 5 was skipped
    print("  Zte_hat_mlp not found, regenerating...")
    model.eval()
    with torch.no_grad():
        Yte_hat_n = model(Xte.to(device)).cpu()
        Yte_hat = unnormalize_targets(Yte_hat_n, Ymu, Ysd)
        Zte_hat_final = Yte_hat.view(-1, *Zte.shape[1:])
        print(f"    Test latents: {Zte_hat_final.shape}")

print("Ready for decoding!")


# ## Decoding Test Latents to Images
# 
# We decode the **test set** predictions to pixel space for use in Notebook 5's hybrid reconstruction. The VAE decoder converts our 4×32×32 latents to 3×256×256 RGB images.
# 

# In[ ]:


import gc

# Step 1: Ensure test latents are on CPU
print("Preparing for decoding...")
if Zte_hat_final.is_cuda:
    Zte_hat_final = Zte_hat_final.cpu()
print(f"  Test latents: {Zte_hat_final.shape} on {Zte_hat_final.device}")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Step 2: Create output directory
TEMP_DIR = SAVE_DIR / "temp_decode"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Step 3: Load VAE
print("\nLoading VAE...")
vae = AutoencoderKL.from_pretrained(ll.vae_id).to(device).eval()


# Step 4: Decode in small batches
# decode_latents_sdvae handles unscaling and [0,1] conversion
print(f"\nDecoding {len(Zte_hat_final)} test images...")
BATCH_SIZE = 16
test_chunks = []

for i in tqdm(range(0, len(Zte_hat_final), BATCH_SIZE), desc="Decoding"):
    batch = Zte_hat_final[i:i+BATCH_SIZE].to(device)

    with torch.no_grad():
        decoded = decode_latents_sdvae(batch, vae)

    # Save batch to disk
    chunk_path = TEMP_DIR / f"test_recon_batch_{i:05d}.pt"
    torch.save(decoded, chunk_path)
    test_chunks.append(chunk_path)

    del decoded, batch
    torch.cuda.empty_cache()

# Step 5: Cleanup VAE
del vae
gc.collect()
torch.cuda.empty_cache()

print(f"\n✓ Decoded test set saved to: {TEMP_DIR}")
print(f"  {len(test_chunks)} batch files created")
print(f"  Output format: [0, 1] RGB tensors")


# ## Saving Model and Artifacts
# 
# We save the model weights, normalization parameters, and test set latents. The decoded test images were already saved during the decoding step above.
# 
# This allows you to regenerate predictions for any split if needed later.

# In[ ]:


print("Saving model and test latents...")

# Ensure directories exist (in case Drive was remounted or paths changed)
LATENTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Save test latents (for Notebook 5 img2img pipeline) ---
if 'Zte_hat_final' in globals():
    latents_dict = {
        "test": Zte_hat_final,
    }
    # Add ground truth if available
    if 'Zte' in globals():
        latents_dict["test_gt"] = Zte
    # Trial IDs for matching
    if 'Tte' in globals():
        latents_dict["test_trials"] = Tte

    latents_path = LATENTS_DIR / "test_latents.pt"
    torch.save(latents_dict, latents_path)
    print(f"Saved test latents: {latents_path}")
    print(f"  File size: {latents_path.stat().st_size / 1e6:.1f} MB")
else:
    print("Test latents not found - they may have been deleted.")

# --- 2. Save model and normalization parameters ---
if 'model' in globals():
    model_dir = SAVE_DIR / f"subj{ll.subject_id:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mlp_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_dim": 15724,  # NSD fMRI voxels
            "out_dim": 4096,  # 4*32*32 latents
            "hidden_dims": mlp_cfg.hidden_dims,
            "dropout": mlp_cfg.dropout,
        },
        "normalization": {
            "Xmu": Xmu if 'Xmu' in globals() else None,
            "Xsd": Xsd if 'Xsd' in globals() else None,
            "Ymu": Ymu if 'Ymu' in globals() else None,
            "Ysd": Ysd if 'Ysd' in globals() else None,
        },
        "config": {
            "subject_id": ll.subject_id,
            "img_size": ll.img_size,
            "vae_id": ll.vae_id,
        },
    }, model_path)
    print(f"Saved model: {model_path}")
else:
    print("Model not found - it may have been deleted.")

print("Saved model and test latents!")
print(f"Decoded test images are in: {TEMP_DIR}")


# ## What's Next?
# 
# We've created and saved our low-level reconstructions for the test set (~1,000 images). They don't look like much right now—just blurry approximations of color and layout—but they capture the essential low-level structure that the brain encodes.
# 
# In [Notebook 4: High-Level Pipeline](https://colab.research.google.com/drive/1sRib_g6UcE6khgj3Ho_4dMWG9oP-4GUP), we'll build a **high-level (semantic) pipeline** that predicts CLIP embeddings from brain activity. CLIP embeddings capture meaning—they know that dogs and wolves are similar but distinct, that cars belong on roads, that faces have eyes. By combining low-level appearance with high-level semantics, we'll produce reconstructions that not only look right but show the right content.
# 
# The low-level reconstructions we just saved will serve as the starting point for **img2img diffusion**—we'll noise them slightly and then denoise with semantic guidance from CLIP. This is exactly how state-of-the-art methods like MindEye work: parallel pipelines for appearance and meaning, combined through diffusion-based image generation.

# ## Source: Notebook4_HighLevelPipeline

# # Notebook 4: High-Level Pipeline
# 
# In `Notebook 3`, we built a low-level pipeline predicting **Stable Diffusion VAE Latents**. These latents capture the *spatial structure* of images (where edges and colors are) but aren't explicitly aware of semantic content.
# 
# In this notebook, we move up the abstraction ladder. We will predict **CLIP Image Embeddings** from brain activity. CLIP (Contrastive Language-Image Pretraining) maps images and text into a shared high-level semantic space.
# 
# ## What You Will Learn
# 
# By the end of this tutorial, you'll understand how to:
# - Extract semantic representations from images using CLIP
# - Train models that predict these representations from brain activity
# - Evaluate semantic decoding using retrieval metrics
# - Visualize what the brain "thinks" it's seeing through image retrieval
# 
# ## Low-Level vs High-Level: What's the Difference?
# 
# | Aspect | Low-Level (Notebook 3) | High-Level (This Notebook) |
# |--------|------------------------|---------------------------|
# | **Target** | VAE Latents (4096-D) | OpenCLIP Embeddings (1024-D) |
# | **Captures** | Colors, edges, spatial layout | Objects, scenes, meaning |
# | **Metric** | MSE, SSIM | Retrieval Accuracy |
# | **Output** | Blurry but structurally correct | Semantically correct but different image |
# 
# Think of it in a more classical Neuroscience way: low-level tells you *where things are*, high-level tells you *what things are*.
# 
# ## Pipeline Overview
# 
# 1. **Data**
#    - Load NSD fMRI responses (voxel patterns from the **nsdgeneral** ROI) and pair each trial with the image presented on that trial.
# 
# 2. **Semantic targets**
#    - Encode every stimulus image with a frozen **CLIP vision encoder** (OpenCLIP **ViT-H/14**) to obtain a **1024-dimensional** embedding.
#    - These embeddings serve as our supervised targets: a compact representation of high-level image semantics.
# 
# 3. **Brain-to-embedding models + retrieval evaluation**
#    - Train models to map brain activity into CLIP space, learning \( f(\mathbf{x}) \rightarrow \hat{\mathbf{y}} \).
#    - We compare two decoders:
#      - **Ridge Regression**: a linear, interpretable baseline.
#      - **MLP**: a nonlinear model that can capture more complex voxel-to-semantic relationships.
#    - **Retrieval test (core metric):** given \(\hat{\mathbf{y}}\), identify the correct stimulus image among thousands by nearest-neighbor search in CLIP space (typically using cosine similarity). This directly measures whether the predicted embedding preserves semantic identity.
# 
# 4. **Downstream reconstruction**
#    - Use the predicted CLIP embedding as a semantic conditioning signal for image generation, aiming for reconstructions that match the ground-truth stimulus at the level of **semantic content** rather than pixel-level fidelity.
# 
# 
# Let's get started!

# In[ ]:


get_ipython().system('pip -q install diffusers transformers accelerate webdataset braceexpand')


# In[ ]:


# Imports
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import webdataset as wds
import braceexpand
from torchvision import transforms
from PIL import Image


# Define root path based on environment
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        # Use a specific Drive folder for results
        ROOT_PATH = "/content/drive/MyDrive/NSD_Results"
    except ImportError:
        print("Google Colab detected but drive module not found.")
        ROOT_PATH = "./results"
else:
    # Local execution
    ROOT_PATH = "./results"

# Ensure directory exists
os.makedirs(ROOT_PATH, exist_ok=True)
print(f"Results will be saved to: {ROOT_PATH}")


# In[ ]:


# --- Reproducibility ---
import random
import os

def set_seed(seed=42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# In[ ]:


# --- Data Loading Functions ---
def build_nsd_dataset(subject_id, split, batch_size, seed=42):
    """
    Creates a streaming WebDataset pipeline.
    seed: Random seed for deterministic shuffling (train split only)
    """
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

    # Use shardshuffle=False for deterministic shard ordering
    dataset = wds.WebDataset(urls, resampled=False, shardshuffle=False)

    if split == "train":
        # Shuffle with deterministic random generator
        rng = random.Random(seed)
        dataset = dataset.shuffle(100, rng=rng)

    dataset = (
        dataset
        .decode("torch")
        .rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy")
        .to_tuple("voxels", "images", "trial")
        .batched(batch_size, partial=(split != "train"))
    )
    return dataset

def voxel_select(voxels, mode="mean"):
    """Averages voxels across repeats or keeps single trial."""
    if voxels.ndim == 2:
        return voxels
    if mode == "mean":
        return torch.mean(voxels, dim=1)
    return voxels


# In[ ]:


# --- Config (using dataclass exactly like NB3) ---
from dataclasses import dataclass

@dataclass(frozen=True)
class HighLevelConfig:
    # Data (matching NB3's LowLevelCfg structure)
    subject: int = 1
    seed: int = 42

    # Streaming (exactly like NB3)
    batch_size: int = 64
    num_workers: int = 4  # Increased to 4 for faster downloads (like NB3)

    # CLIP
    img_size: int = 224  # CLIP ViT-H/14 also expects 224x224
    clip_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # SOTA 1024-dim OpenCLIP

    # Ridge parameters
    ridge_alphas: tuple = (10000, 30000, 60000, 100000, 300000)

    # MLP parameters
    mlp_hidden_dims: tuple = (2048, 2048)
    mlp_dropout: float = 0.3
    mlp_lr: float = 3e-4
    mlp_weight_decay: float = 1e-4
    mlp_epochs: int = 150
    mlp_batch_size: int = 256
    mlp_noise_std: float = 0.05
    mlp_cos_weight: float = 0.5
    mlp_softclip_weight: float = 0.5
    mlp_softclip_temp: float = 0.07
    mlp_patience: int = 20


    # root: str = "/content/drive/MyDrive/NSD_Results" # We set this dynamically above
    root: str = ROOT_PATH

hl = HighLevelConfig()
print(hl)


# ## 1. Data Loading
# 
# We use the same streaming approach as Notebook 3. Images are stored as uint8 to save memory.

# In[ ]:


# --- Main Loading Function ---
def get_dataloaders(cfg):
    print(f"Setting up streaming dataloaders for Subject {cfg.subject}...")

    train_ds = build_nsd_dataset(cfg.subject, "train", cfg.batch_size, seed=cfg.seed)
    train_loader = wds.WebLoader(
        train_ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=True
    )

    val_ds = build_nsd_dataset(cfg.subject, "val", cfg.batch_size, seed=cfg.seed)
    val_loader = wds.WebLoader(
        val_ds, batch_size=None, num_workers=0, pin_memory=True
    )

    test_ds = build_nsd_dataset(cfg.subject, "test", cfg.batch_size, seed=cfg.seed)
    test_loader = wds.WebLoader(
        test_ds, batch_size=None, num_workers=0, pin_memory=True
    )

    print(f"Loaders ready. Batch size: {cfg.batch_size}")
    return train_loader, val_loader, test_loader

def take_n_samples(loader, n_samples=None, seed=42):
    """
    Takes n_samples from a streaming WebLoader.
    Stores images as uint8 at native size.
    """
    set_seed(seed)
    xs, ims, trs = [], [], []
    n = 0

    pbar = tqdm(total=n_samples, desc="Materializing samples", unit="samples")

    for vox, img, trial in loader:
        if img.dtype == torch.float32:
            img = (img * 255).clamp(0, 255).to(torch.uint8)

        b = vox.shape[0]
        xs.append(vox.cpu())
        ims.append(img.cpu())
        trs.append(trial.cpu())
        n += b
        pbar.update(b)

        if n_samples is not None and n >= n_samples:
            break

    pbar.close()

    if not xs:
        raise ValueError("Loader yielded no data!")

    X = torch.cat(xs, dim=0)
    I = torch.cat(ims, dim=0)
    T = torch.cat(trs, dim=0)

    if n_samples is not None:
        X = X[:n_samples]
        I = I[:n_samples]
        T = T[:n_samples]

    perm = torch.randperm(X.shape[0])
    X = X[perm]
    I = I[perm]
    T = T[perm]

    return X, I, T

train_loader, val_loader, test_loader = get_dataloaders(hl)

print("Materializing Training Data...")
Xtr, Itr, _ = take_n_samples(train_loader, n_samples=None, seed=hl.seed)

print("Materializing Validation Data...")
Xva, Iva, _ = take_n_samples(val_loader, n_samples=None, seed=hl.seed + 1)

print("Materializing Test Data...")
Xte, Ite, _ = take_n_samples(test_loader, n_samples=None, seed=hl.seed + 2)

print(f"Xtr: {Xtr.shape}, Itr: {Itr.shape}")
print(f"Xva: {Xva.shape}, Iva: {Iva.shape}")
print(f"Xte: {Xte.shape}, Ite: {Ite.shape}")

# Voxel Handling
# Keep raw training voxels for MLP expansion, average for Ridge.
# Val and Test always use averaged voxels.

Xtr_raw = Xtr.clone()

Xtr = voxel_select(Xtr_raw, mode="mean")
Xva = voxel_select(Xva, mode="mean")
Xte = voxel_select(Xte, mode="mean")

# Expand repeats for MLP training
if Xtr_raw.ndim == 3:
    N_img, R, V = Xtr_raw.shape
    Xtr_exp = Xtr_raw.reshape(N_img * R, V)
    print(f"\nRidge training:  {Xtr.shape[0]} averaged samples")
    print(f"MLP training:    {N_img} images x {R} repeats = {Xtr_exp.shape[0]} samples")
else:
    Xtr_exp = Xtr.clone()
    R = 1
    print("\nVoxels already 2D, no expansion possible")

print(f"Val: {Xva.shape[0]}, Test: {Xte.shape[0]}")
del Xtr_raw


# ## 2. Target Generation: CLIP Embeddings
# 
# Unlike Notebook 3 where we used VAE latents for spatial structure, here we extract **CLIP embeddings** for semantic content. CLIP maps images into a 1024-dimensional space where semantically similar content clusters together, capturing **what** is in an image rather than **where** things are.
# 
# ### Understanding CLIP: Contrastive Language-Image Pretraining
# 
# CLIP learns visual concepts from natural language supervision. Given a batch of images and their text descriptions, CLIP learns to predict which image goes with which text.
# 
# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAJOCAIAAACm/eLyAAAAAXNSR0IArs4c6QAAIABJREFUeAHsfQd8VMXa/pyzG/Sq/3u9ev3utV4VP/30igUQBUX03mvBElRKEkrohGLo1VBFlF5FKcluKAKRXgQpoQdIQkIqSUjvhfS22XLO+Wf3Ia/HTU7IJgECnv3tbzNnzjvvvPPM2c3zzrwzw6Tb5CUqvG4T81UzbzECCo+PYvaNNlepYqV6leTVfBUBFYGWg4BQ82ohJtWYIzTRnubS00QzHC1+XbOvK+BojQ2UV6pXKb+BakmsufSQwvoTN7m6+o1pyN2WZrDS//2m59ePBmt6BaoGFYGWj4DS10DJckfllfQo5TuqX0lezVcRUBFoOQiAWNx59rS0djUc4fotr/9uw2txVFKpXqX85tLvqJ6GyzeX5Q2vsSmSLc1aJZ7Q9Pz6UVIdgKYjrGq4AxFQ+trcqqYq2fNHy5ckqeFNdki44WqbXfJ2sbPZG26nsLlwuIV6BKGpY+12mLSQy9uiXUr9Xo/xSreUVDnUHfUoUarXIf31CCvpr8ekerTZ3VJSolSpXXG6VNJDAnYJR+XtissvHTVVXvZGpG8cr6jfWtUBuHHIq5rvQAREUbwlrar/a/zHuUvg19/kBorVr+Qm3L1d7LzRUDQXDi1Nz43GrRH6AVEjCrbkIs3V79TGJipsYnEyg/7dyHOakm6iYU0sTpY7qsdRearodklQA5s9UT8CqgPQ7ICrCu9kBOjrdJMbSfX+wRNy2JWgaIiMUtmbmd8UO+Vl5Wkl++Uy8rSS/M3Mby57WpqepmAob4s83Vw6m6KnpZVtLnzk7WqKzqaUbS4b5Hrk6abY1pSyTbGhueqV29DS0vI2Nm+6/pY67AAoqWsuox3VryTvaL6S/Up6lOQdzW8u/c2lx1H7m0teyf7myleyU0l/Q+SVZFpCvqPtagk2O2rD7dhGuc2OtvdOkm8uHFqyHrLtFnYcbGi6AdQWeaIetXKxhqSVVDWibJ1FHNJvJyxXaHeLLhsiQ8JKoSyUL5dsXFpuD6XtVFE+zTPYCZA9kLS7Ky9OaTsZuiQBu7rk+UppUtKMiTrrsmuvXKYZq77JquStqJ1WdABqi6o5KgK3LwJY9NP0TzkCN/mb7FB1cjvlaYeUtHxhedOQbuE2k8Et3M4bbV5z4dBi9TSXYTe6Ixqon5pDifoLklgDE0rarlu8zoK1S9UpVifhq1OSFNZ5V65HScAunxRSwk6giZeklhJ1Kqz/7nXbRcUpUWct9eihgkoJJYVNzK9dHRTWzkdOE6u7hcWVWoT828YBUKJu9TdPvasiAAQsDr7MCi87PG/hF7v+qu3spMv6S92Od6lpdmNLLbMtsLZl2nYzrWouHFqmHnombyakN7QualEDAbeTv+6lkvH1F1QqJWec9Rtsp/+6CpUE6q+ldqmG11u7bENyGq6/fsvrv+sozrUtt7PT7rK2fHPlKFVkl0+XzVXvzddDTagzwZCrZJYgCJs3b546deoM22vatGleCi8IzJgxY3rNa+bMmbNmzaq5sv+roMbhbNTo5eX1le3l5eVlX1PTrh01qGm13bzSM2fOnDNnzsyZM728vGbMmOHl5TV16tT6Gzt9+nT08vTp07/66isvL69Zs2YB+Wk1L2ibOXPmtGnTpk6dOmXKlK+++mrGjBmzZ8+eNWvWTOWXoy1XMlVJj5I8Hht6eEhMni9PT58+HTIzZsz4+uuvZ86ceenSJYvFovQNulX5dX7b66HISvJK+Y62q/7fGUe1NVy+ee1X0kb51zWMJO0SSgXtxK57qaSnufKva0DjBGqb10A9tQve2hwlsx21SkmPUn5z6b/ReprLfkftvN3lHcXNUfkbjU9z2dMy9dxo9BqhXwkou/zrOAAlJSUTJkzQarVDhw4dNmzY4MGDhyi8BtW8Bg8ePLTmNWTIkJps+7+DFV4DFV725Wuu+/fvP3DgQKoUNQ4cOLDmflP/KpipiIMCPIo4NNE+JfMoX8keDw8PYIU+BWLoYiorTwwZMoR6f9CgQVQK+YMGDYIGdMQw24t6f8iQIXgiqJtqHpDf/irZ6Wi+Ep7ytjQlPWLECA8Pj2HDho0ePbpbt26MsQMHDjTi+3mji9h9z+lSqV4SaGBCSY9SPtQq3b1x+UrNcbRGJT12+ddVaydPl0oFSaCBCSU9zZXfQDMcElOyrSFKlMreqnwlmx21R0mPUn5z6b/ReprLfkftvN3lHcXNUfkbjU9z2dMy9dxo9BqhXwkou/zrOAAGg2HChAkjRowwGo2VlZWCIChERpiNRqPBYKisrKyqqjKZTGaz2WQyGY1Gk8KrHj3Gul4KaqxVyGsxGo1VVVUGg0FJ3tF8JTuV8pUiTRytt7nkleyRJKnK9rJYLCaTCR1nsViU2mU2mw22F9CG2qqqKsiTtWaz2WKxCIIgSRKeFsiTWiV7HM0nhXYJssQuYSfW6EtBEICYJElnz57lef748eP1jKw34qt7RxbB786d1zS739M7r4E3qEVy3JSqaIiMUtlblS+3WZ6+Vfao9aoIqAjcOATwHb9x+hutWf7jU09a0QEAjSsrK5swYcLIkSMR7HVda0RRxAkL15VUBW45AmazuVmYq93jJW8X6LL8kbATpkt5qZuZJgMaCIUoinAeJEk6c+YMY+zo0aNQcjPNvu3qulMhkj8/DXyEbru+uxEGE271K2+gWP1KbuZdMtgucTNtUOtSEVARuDkI4Gt+c+pyqBa73x+lS0UHAJWVlJSMGzfOw8NDzuHqt0MURRpIric2Wsmg+pXXvktHOte+1Sw5zWVnsxjTvEqoTy0WC3EXpfZiUJ/8QIjRGiBKwAOUC9PqbRivpL95m1Zbm1K98vzapWrn4PFG/tGjR3meP3r0KFwpwqR2KTXnTgVH/vzQl0jt7usi4BBWDglft+obKmD3PNDlDa1UVa4ioCJwSxBosf/X6Jen/sR1HIDy8vJJkyZhBsAhfDH0azabHSrVdGG0lgho0xXe2RrIW7suYuTLXXeSBwJ2uLXY74mdnde9BBeBy3T69GmEAMGVom/adZX8AQXumAfAru+o0+/UBtq1V71UEVARUBFQEQACLfZn3+4fk9JlfQ4AYp3HjBnj4eEB/mcymerveISAy2WUKpbLyNNK8kr58rKUVhJuRD7pbGKiEVU3SxEls0k5CVx3mI18AAT3I8qfiiNBHoVdfNF13QY7PY2+pHbZJRqtsM6CaF1AQABj7MiRI6oDUCdK8kx0hzznzkjf0MfszoBIbYWKgIqAisAdiUCL/b9m949J6bI+B0AQhIqKigkTJowaNcou9qO2OiwqLbC9Kisrr9vZtTUg57oF7QRycnIKCgoqKyuJv1osFqPRaCfW6EslOx3Nb7QBdgXl9drdqvNSLl9nGqVKSkqysrJycnLqlBFF0Wg0pqWl5efnS5JkMpkMBgMKUhwRnhA4jUajsby8vKqqijqFnp86jZRHECkZ0MD8m6DfZDKhXSdOnOA47vjx43YOgLzVSvb80fLRfXdeq+0eyzuvgWqLVARUBFQEVATqRKDF/l+z+8ekdKl4EBhaW1FRMX78+BEjRlDjBUEAvYZGGhWWJOnUqVMcxzHGDh06RME/FFtCCbpFOkGYoAqfRCtBIqkI9oqRJAk5ubm5HMcNGjSoqKiIWJfcJDBLWrUpD+BG7QhVwmA2YYR5DNKDKHZJkpCDqtEc5KBqwCIIQnBwcFRUFPZBwpwJ7RtDqlA7IknkrSMqTHDZUWfURWE2FGEvj+OnpskRrt1faDts27p1q1ar3bRpk8FgsLMKSn755RfG2Ndff004kHJ0FgwWBKGsrMzNzY0xlpGRIZ8loBZhHS1dQo/ZbLbLQT6ppUs0GUbK8aQHgBJUOz02ZDN6h+YlsGUQ7hLC8ku02s7CkydPMsZOnDhBam9+At168+utXWPLsaS2bU3PoR8Hu0TTNasaVAQagsCd/f1qCAKqjIrA7Y7ATfsW2/2fUrp02AGQ81E5VSotLZ0wYYJGo2GMeXl5lZSUoKuIyhMbk3chcTU5k7OjWXJ5pInQp6WlcRw3ZMiQwsJCMgY1Go1GtBn0GuuSiV4T3SQHQF4LWUU+gNFoJIYtb5G8FBqYlZXF8/wnn3xCCIBl2jlOGEeXexRErKn5qIiMoVajUsrHJZlKdkqSBJ+EHAxqNdUFQMxms16vZ4xt3boVzBiwwBJUtG/fPo7j5syZQ+aRGXatKCoqatWqFc/z8fHxVDWUEBRUFqBBp9FolCuvzdrl9kAhciwWS1VVFdxC1GgHDpopD2AjxwmWyJ9qypHbJs+ktOoAEBT0zZLn3ElppR/QO6mNaltaMgJ4AluyhaptKgIqAvUjcNO+xUr/sOzyG+MAoIXUEpCtvLw8xthTTz1VPU7M83xcXBwNmsrJPVgaSGptlianaLSzCqojqifHNy0tjTE2dOhQhKaAsZFhxHflFYFV1+b9chli5/AiSJgSchtoOgKZGRkZ1aPCXbp0oSAZIsGQFEXRZDKBsBLfJacC4/E0wA+rSIzyURfFokAnmk+22fFy4scYgJczNlEU4QD8/PPPdJf0EGuPj4+nUCu5myF3J8Dak5KSUlNT5Q0nMl27H9FkbDsLm+2ov7wuOb+XPy1krSAIBAsy6VwCOzsNBgN1eu1HlB4eAh+9T0UkSVIdAIJd/jjJM++YtN3vJl3eMQ1UG9LCEaD/ay3cTtU8FQEVASUEbtq3mP5D1Z9opAMApSBJIK9r165ljG3btm348OEYS7Yjr3JEcnNzw8PDS0pK0tLSzp8/Hx4eTmd/paamnj179tixYxcuXMjMzKSKMIQvCEJUVFRwcHBISEh6enpOTg5jbNiwYVevXoV+MEji06IoJiUlXbhw4fTp0xcuXEhJSaGx3vLy8suXL5eVlZWWlgYFBV28eDEoKKiwsFBupyRJFRUVUVFRZ86cOXTo0MWLF8vKyogRGo3GiIiI1NRUQRBiYmICAgLCwsIuXrzIcVzbtm1PnTqVkpICEEpKSoKCgk6dOnXs2LFLly6hFvBaOfE1m81wDAoKCoKCgo4ePXrq1KnIyEiYRNFEV69eDQkJOXny5Llz5yIiIgoKCsjm8vLyiIiI3Nxck8l06dKlixcvQiEEiOZidUdISEhAQMD58+fz8vK2bt3KcdzmzZsFQSgvLw8PD4+OjhZF8cqVK+fPn8/Ozi4pKQkPD8/IyKisrIyKigoLC6uoqADaaGNZWVlwcPDly5fNZnNqamp0dDT8qMLCwsDAwOLi4oqKivDw8CDbKy0tDR1BJqH4GdsrIyOjsLAwKioqJycHnoOdL4HmZGZmAqWAgAD0LPJFUczOzg4ODq6srETnBgYGBgcHFxYW1nY/6HnAyubo6OgTJ04cPXo0KCgoISFBTvcp8g21SJJ04sQJNQSI0MBXlS7vsITSz2hzNVP+pDWXTlXPnYRA/d8v9fm5k/pabcudikD93+JmbLXSPyy7/EY6ADAUB8GCQnXu3BkD/6dPn2aMPfzwwxRwj98marnJZFq9ejVjbPfu3SNGjGCMcRxXVFRkNpvPnz//ySefYA6BMebm5nbu3Dn5wO1333338ssvI9Coa9eu3t7eSg4AFiX7+fm98cYb1bErHMfxPP/hhx+eOnUKxp85c4bjuFWrVo0dO5bVvIYMGRISEkLdcPXq1Tlz5mi1Wp7nYdWYMWMyMzMxzBwfH//kk09+8803R44ceeyxx7RaLWPs0UcfxUIIxtjcuXMxNj9p0iQo4Xleq9WOHz++rKyMVhoQuwU9vXz58uDBg0kJY2zNmjXE42NjYz08PKqVAwSNRuPp6ZmWlgYufuHCBScnp0WLFvn6+vI8/9xzzyUnJ1NHoF1ms7mkpGTq1KmtWrVycnJijA0ePHj8+PHw3yRJSk5O7ty5c9u2baOiol577TXG2Pnz59Gtq1evNhqN/fv3Z4ydPXsWCtG/27dv5zhuxIgRZWVlo0eP5jiuvLwcw+ROTk7Lli1bsGABGsXzfLdu3S5cuIAnx2w2x8fHT5w4EU8COnTevHkcx+3YsQMOAJwluECA4siRI87OzhqNhud5juM6dOiwd+9ek8kEDHfu3MkY27Jly1dffYV+YYz1798fyxLkgNDDaTAYlixZ8sQTTwBYeHF79uwBsASd/B8tHIDjx4/TA3PzE/S1uvlV29XYciyxM6xZLu1+N+myWZTji3BnA9hcQP1h9dT/eDT7A/mHxVltuIrAjUOg/m9xM9ZLPwj1JxrjANCoLcwVBAHscMSIEdnZ2WazGUwuKChI3h6KJJEk6euvvya25+bmtn///vLy8vPnz4OHz5s3b9OmTdOnT8dlWFgY/kF+//33jLHnn39+7ty53t7eLi4uIJRKIUDHjx9njD399NNbt249fvz48uXLGWNOTk6JiYmiKJ49exY0um3btqtXr964caObmxuoZFZWFgLoP/zwQ8aYi4vLhg0bdu3a9eWXX2q1Wo7j8vLyJEmKiop66aWX7r//fp7n33nnHT8/v8OHD69btw6Vrl27NioqqjrwxsvLS6PRjB071t/ff//+/f379+c4bsGCBXIaSmsMsKyZ47i+fftu27Zt5cqVbdq04Xl+y5YtkiQVFxc/9thjWIx7xvby9PSsJugffPABopVOnToFYDUazccffzx9+vTc3FygZzAYMMhdVVXVr18/nuc7duw4f/78RYsWdezYEe6Nr6+vxWLJyMjo0KEDsH3qqadGjhxpMBgQ7rJ48WJJklatWsUYmzlzZnFxMbln0BAbG1tSUtKjRw/GWHZ2tiRJx44dg0kvvPDCokWLtm7dOnLkSMZY586d09PTLRZLenp6z549eZ7v1auXj4/P/PnzX375ZRTZsmULrdYlbwFOBc/zGo1m0qRJGzdu/Oabb+677z6sPscjt2PHDtj/5ptvfvfdd8uXL3/vvfeqc2bPni2fHZKHby1cuFCj0bz11lvbtm07d+7cDz/8wHGcRqPBVAZqp0/UojoAwAGfN+2nTV7pTUsr/Yw2lwFy/c2lU9VzJyFQ//dLfX7upL5W23KnIlD/t7gZWy3/Qagn7bADIA/IxmioxWJZvXo1x3G7du0Cb166dCljDJuH2jUJgdQzZ87EuOwvv/yCfX6qqqrAIA8cOEA0a+vWrYyxTz75RJKkpKSkjh073nPPPSdPnkS9V69exUh5nYuABUHo1auXk5PTzp07MYhrMpnmzp3LGEtISJAk6cyZM9WU8fXXX09ISMCwcUlJyVdffcUY27RpkyiKhw4dwnLeoqIi0FyLxTJlyhTG2JIlSxCM1K5dO8bYtGnTMC0gCEJVVRWmGkpLSyVJiomJadeu3WeffZaXl4eIqeLi4n79+rm7u5eWlsIwOFSIMFm5ciVjbMKECSguiuKBAwcYY+7u7kVFRRERERzHjRs3DuPi4K/ADVuv+vv7gzqvXLmywvaCm4Few5D8/v37tVrtRx99hHUakiQlJCS8+eab1fM2a9euNZvNaWlpb731lpOT08CBAyMiIvD0HD161MnJaf78+aIoFhYWghxnZWWhCSkpKRqN5oMPPsjLy6sOmurRo4dGo4EDcOjQoerB+GeffRZD/pIkFRQUtG7dmjEGbr1//37G2Oeff15cXIyeDQ8Pb9u2rVar3bp1K1YImEwm4IOOcHZ25nkeAUtAD+4cz/MFBQWiKO7YsUOr1X722Wepqamk8+mnn3733XfR+/K1BHhEMfBPmFTXi9kDvV6PxwOPJbShiOoAyL/dN+2nTV7pTUvX8xvq0C0lg+VKlGSaki/XL087qlNeVp52VE9zycttkKebS3/L0SNvXf3plmCzkoUtwbabaYMSDkr5N9O2G1GXUruU8pVscFReSU9Ly0e7boJVSgDa5TvsAMB0WqYpimJKSspHH3304osvYmRdkqSIiIjqA5I6deqUl5cnX7dK0RQzZ85kjK1YscJoNIIWh4aGarXagQMHUhSH0WjMzc1t3bo1z/OSJP36669OTk6TJ0+muCNJki5cuKC0DagoihhpHjZsWHx8PMwuKSlJSkpCJMnhw4c5jps6dSq4IEge2PasWbOMRmOfPn0Q4Q3OB+PDw8PhNpjN5ujo6DZt2jz33HMY6UcVV69e5Xm+a9eu2JgoKiqqffv2Tz/99L59+6jXs7Ozs7Ky5IwWt8rKyrCC4vDhwwigx6j/mDFjNmzYUFZWlp+fv337dkTGw6qKigqEJ8Fh8Pf312g0vXv3xsC/3E/ANkSVlZU//vhjdajS+vXrUSmav3TpUq1Wu2zZMlEU09PTO3XqhMgfWveMuYVly5aBgsN/O3bsGMzAGuINGzZIklRUVOTq6soYgw0HDx7kOO67777Dqmjg3K1bN47jQkNDDQbDmDFjEA+GqC2LxVJZWbl48WKO42Bkpe2FR0WSpLCwsAceeMDNzS09PZ3mHwRBePnll3mex3zRtm3bNBrN2rVr6fErLCx0d3d/5plnYmNjYTM+idxjjQcwwYKTzZs3I/6KPFJ5gtYAqCFAAO2m/bShupv8afe72ehLJbPtFCqJNTrfTj9dOqqQCtolHNXTXPJ2ZtBlc+lvOXqoaQ1J3HKzlYy85YbdZAOUcFDKv8nmNXt1Su1SylcywFF5JT0tLR/tuglWKQFol6/oAIAelZeXjxs3buTIkUQlazcAQ+nvvPPO4cOHDx48eOjQob17977xxhuMMTnrpUH0ahYFB+DQoUMExOrVq3mef/3112fMmDF//vxZs2Z9++23iG9hjBUWFu7du5fn+WXLloGSgrcVFxdrtdq+ffvKdwGikdqMjIwHH3wQ0f8dOnTo0aNHQEAAEb5ff/2VMebr64umwSsIDg7mOK5Xr17VC0w/+ugjjuMoAgSMPD4+/umnn37qqacMBkNkZORjjz0mX4IsiiJieN5//30sFxYEYf78+QhHeeGFF4YOHbp3715YK1/+CxtSUlL+85//tG3bNjIyUhRFivs3GAwwDxhmZWV9//33ffv2/fe//w32Xz0tgF1Hz549y3HctGnTEJwD+ouKcJjX1atXp06dyvP8wYMHcRcd7efnx3Hc4sWLKyoqMjMz3377bZ7nq6qqML5usVj8/f15nl+wYAHg3blzJyY6RFGsqKgYO3ZstdsTGBgoSVJpaSnOAYALdODAAZ7nfX19qQmiKLq5uWk0msDAwLKysj//+c8cx9HQO8Q2bNiAIH74XWazmU6X279/v0ajWbJkCR00hoZ888031cFXGzduNJlMvr6+mMmhB6yqqmro0KEPPPDAxYsXkUnuKMnk5+dv2bLl888/79y58wsvvIA5AR8fH+L9eHiAmCRJiDE7duwYPXJ23y6SpCoakaj9jWuEkj9skdo9ouIpfxhaGj6O2uOovLztt2Pa0fY6Kn+rMIGdDa/dUfmGa3ZUsnktuV36SwklJfuV8pX0tPz85u33Zm+vEuB2+Q47AETIQG4MBsPkyZMRrA+qfdddd9Hl7Nmzy8vLqUoiUoi0OXv2LA3rzp49G6EsL7/88lNPPfW3v/3t8ccfb9269SOPPNK1a9fi4mKEA61atQpIwYzS0lKEx8j31ZH7KikpKTqdrkuXLhqNBkuBR4wYgZFphIljmSnpvHLlyj333OPs7BwfH//hhx9yHEeb8IAKJyUlPfvss0899VRFRUV0dPRjjz02fPjwnJwcYpOZmZkIW0IsCmj3xo0b3d3dQdYRonPgwAEAiE8Uj4mJef3117t06RIVFYWCxCxhYVFR0bx58wDvf/7zHy8vL51Od//99zPGEOIfHBzM8/zkyZMR7UNbkWKFgMlkysjIwCQD8WBsmompj3nz5mHi5Z133mGMkbMkSdKRI0cYYwsXLoQlqamp4MfYFOjhhx8eOXIknJCysjLMAOTl5QmCAEfLx8eH5naqA7rc3d0ZY2FhYejB6oXIGM6HcrPZ7OfnRwweZmDhtSRJu3fvJleQ5kkEQViyZAnHcStXrpQkydfXl+O4bdu2QSFipYYPH96qVatLly5RJsCHfqhljL3wwgvDhw9ftWoVpoC2bNkCMVqrjUvVASAYW3KCfnzsEi3Z5ptpmx0sdHkzbZDXRQbYJeQy8rSdGF3KZe6kNDXQLqHURjsxulSSv1X5MKzhtTsq33DNjko2ryXUQXYJR626VfJ2Zl/38lbZ2fR60bSm67lBGq6LPAQcdgDk9Brh4OCjO3bs8Pf3P3z48L59+/z9/RET8pe//IX2UgTHxTbziK729/en47p8fHwYY6tWrUpNTU1KSkpOTo6Pj4+NjU1ISCgqKrJYLH5+fjzPL1myhPAym83FxcXVi009PDzy8/Mxzo27RNGwhDQrKys2Nvbo0aMwdeXKlSaTac+ePYyxzZs3k0JJki5dusQY69evX2pq6meffYY4dWJ+kiTFxcXxPN+mTRuTyRQZGfnoo48OGzYMa4JRKY4m+PTTT8vKyuSa8/Ly4uLijh07hgW4TzzxBFg+SmFaIzEx8b333nvllVdiYmJofkAQhMLCQqxLxurbLl26xMTEZGZmFhYWlpWVtWvXrjqWCZFRZ8+e5Xl+xowZtEennCIjPmfy5Mkcx506dYpODxBFETMA3377rcFgyM3N7dq1K8dxcjAx2l09OQOfpKysbPbs2RqN5tSpUydPntRoNBRTVFRU1L17d47jrl69arFYDh8+DCpP7oQgCEOGDGGMnTt3rqKiAr5Zamoq4soAyMaNG6sXfPv5+cnX6eKRxQ4/CxYsgJMDeyorK6dOnYoOFQRh06ZNcABwmoQgCAaDYdiwYVqtNjw8nDoUJplMpoKCAjwbu3fvzsjIyM/PF0VRp9OREwIoUJCeruvOAMgfgEanW/gPTaPbpRZsCQgo/Z9oCbapNtRG4E7tL0d/5RyVr41kc+W0HEuaq0WqnoYg0ML7HeZd97NBDgCW8xKBIwJkNpsxMP/ss89SJmH3xRdfaDSan3/+mXLobNo5c+Ywxk6ePElzAkFBQTzPu7i4EKtclZCSAAAgAElEQVSTJCkvL2/lypWenp6SJB09ehQbaCIwBlQMnN7T05P276eReEEQOnbsuHbtWoo8kSQpPT2dMdaqVSuz2Yy1sF5eXiDfULhr1y7G2Pz5881m84ABA6r3gty3b5+c9iESxsXFBd7C008/PWzYMKx2BT6pqamYlKioqJAkKTw8fPz48TgQgMawMWdCq5kpkL24uBjrFvz9/QGaxWKpqKhAwH12djbOWaOoJEEQ4AIxxgAL9jadNWuW/JQr2A+PwmQyrVixgud5eWSLJEnz58/XarXff/+9IAiZmZmffvqpRqPBKgVsm3P48GGNRoNdgBCsjzmBfv36IaALxxHAJCwCRqQT5hb8/Pzo+ZEkCTuZBgcHGwyGoUOH4uAIehiuXr06fvx4juN++ukncgDwgAmCEBAQgC2DsrOzKcapoqICW5pGREQYjcYNGzZwHAf/AU9deXn5gAEDeJ4PDQ2lh4SQp34nt00UxW+//bZ6QgAPMDBEp9CjXo8DIH/mm5hu4T80TWydWvzWIqD0H+LWWqXWroTAndpfjv7KOSqvhGfT81uOJU1vi6qh4Qi08H6Hedf9dMABIN5D+1dWb5Xz+uuv8zy/fft2kEXcAtXT6XQcx/3jH/+g42MJ3BkzZjDGAgICwPmwI/4TTzwBvgUSlp+fv379+uqwk3vuuQcrbrFTza5duygax8XFhTE2ZswY2jOHNnmUJOnhhx/WarX79+8nwoe9LDt37iwIAtJPPPHEgQMHEF9+5coVnEtw8OBBOueVMRYZGWk2m41GY1RUFHbhxLrP0NDQJ5980sPDo7CwkNyMnJwcBMdnZmZWVlYGBARUbyXZs2dPHH0lSVJZWRlqOX36NE2noEWSJG3fvl2j0Xh5eWHI32AwYPfSsWPHFhcXY2XF9u3bDQaD2WzOyMhYsWIF3AM4TgEBAYwxLy+voqIiYszyWrByuhqWdu3aYQZGEITIyMi3336bMQavIDs7+4MPPmCM0VJvSZKwv9CCBQvI1NTU1I8++ggD52+88QYx9aKiom7dulWf0JyZmSlJ0qFDh+yiccgBuHjxosViAY1+7bXXIiMjy8vLs7KytmzZgniwvXv3Ekenp66oqAiHRWzatAluT0VFhY+PD7ZsqqysFAQBSwh27txJEyBVVVUeHh4ajSYoKAjPJz3PgiBgx6fq5S4IW8rLy4uKivqf//kfnudpZoPkKaHkANBz3iyJFv5D0yxtVJXcKgSU/kPcKnvUeutH4E7tL0d/5RyVrx/VptxtOZY0pRVqWUcRaOH9DvOu++mwA0CjwiaTCXSzetN6ivMhumaxWJKSksAOcWKUfP/QadOmVe+dLx8ClyQpICAAXHbChAkrV66k87lCQ0PRNwgTuvfee5cuXbpmzRpnZ+dWrVoxxoYPH44ZABA7ijUKCgpijGm12iVLlnh7e69YsQL2YNUmHTvg5OQ0Z86cdevW9erVS6PRuLu7YwWtJElz587lef6BBx5YuHDhihUrsOnnlClT4GZERUXde++9gwYNQggQ6q2srESsv6enp6+vb1lZGUa4+/bt6217TZgwAYdSYddLlCL/wWAw4HSCgQMH6nQ6HI/FGLt06RJtCarVahcsWLBixQqstcWigkWLFmVkZJw7d66auE+fPh0hQAi7J/8HMFoslvHjx2s0mr/85S+LFy9etWrVc889ByVr166tZufZ2dnwtcy2F/gu/KWvv/4alBpuANbdajSa/fv3QzmWQcMxgA+DbUARakXu2aBBgxAChMmESZMmVR/K9uCDD06bNq1fv34cx/35z3+mg8nIgSHmfeLECYD8zTff+Pj40JERmDkxm81wAGj2SRTFyspKaA4MDJTPReAbUlJSgmdj5MiRy5cvxwQFcv7v//4P/h71EZmhOgCO/miq8i0NAaX/EC3NTtUeIHCn9hfa1fBedlS+4ZodlWw5ljhquSrfFARaeL/DvOt+XscBqKioGDduHEKAiPcQIduxY8dHH330448/YviZmDcEqqqqXF1d33nnnWXLlmFfRQrTX7t27WuvvXb+/HkaoMVUQEREhLu7+9/+9jccqTtx4sSUlBQMLaN2vV7fpUuXe+6558knn+zevXtUVFS3bt2+//57mgGgcWjYcObMma5du4LdajQaFxeXs2fPws7g4GCcUPvTTz+1bt1ao9G8+uqroM5QYjQaKyoqfvrpp06dOmEbnzfeeGPNmjWIdZEkKSUlZfDgwfPnz68+QMBkMmEeQ5Kk6uUQnTt3fuKJJ1avXg1VY8eO7dChA6KP2rRpM3jwYAQIEZKgpIgzKSgomDRpEg48fv755wcMGID4eJPJVF5evmXLlk6dOmk0mkceeeT999/PyMjYsGHDyy+/DDxjY2N79Oixfv162kGIngA86+QjLVy4EHt9Pv744+PGjQsODn7rrbfAmPPz86dMmdKrVy+MpgOuoKCgDz/8cM2aNZgWwOatycnJL730kouLC6J9TCaTxWIpKCiYPHnyG2+8kZqaCr/uv//97/79+ykSSZKklStXvvnmm/L1uD/++OPnn3/etWtXV1fX06dPYw0JLdEm75EewlOnTmEeAHMFn3766YULF2izoJ07d/7rX/86dOgQcf3Kysp58+b16NGD9oSVf/mrqqrOnTvXtWvXBx54gDH29ttv79mzp7S09JNPPrn33nsXL16MM9TgA5ANdToAcrXNkm7hPzTN0kZVya1CgH4f7BK3yh613voRsOsmuqy/VMu/i4Y03E5H5Ruu2VHJlmOJo5ar8k1BoIX3O8y77mdjHAAs5AV2RCjBivBJAdMmk6mkpKSqqgoEDvnEn0B/oUGeGRsbe+HChcTERFBP2i8fjamoqIiJiQEnhv8AsgsN0CbXWVRUdP78+VOnToWFhcnzMX2xYsUKSZIyMzNDQ0MxkA8ZejIsFkteXl5gYGBwcHBOTg7uYmkpySAhb0JBQQHGvwml3Nzc4ODgkJAQ+YIBGhGHGNFcSZKysrIiIiJwvhiiX6gv8/PzIyMjk5KSqDm5ubk4ckFuA1kFGOXD+XC3SktLIyMjk5OTqcm0DBd8FxrIS6EhcAoNku9VCmGSobkgLLGg+H6ymdSWlZVFRETAIyIXcdWqVRqN5tdff6XIH/kjhLrKysoCAwOPHj2KnqVWoMaioiJyyahSu4eEVlojv7S0NNr2wpZKgiBUH2oWFRWFnazoqSaQazsAhFgzJtDvzahQVaUiQAjQr4pdggTURItCwK6b6LJFGdkIY9CQhhd0VL7hmh2VbDmWOGq5Kt8UBFp4v8O8636y+ptRUVExfvz4ESNGACmiPk0BztGy1AZQRpBvrP50VBWRTlEUAwMDOY6bN28e+CJiWqCWWCzVRTZQohFV39AiZFgDE40zhh4Aua/ikCpogA8JzNER69atgx5RFK9cufLyyy8//PDDOFquzhbVUynoPvkSZHM9RRp369ixYxzHIaJM7tyiRsx6kfF2VcBIeIAkj02x4DvJBeA+3biG2Nn2+0tBkq69RUmwvSVRuva+JknX4m+3ZHn1Zdo0UBW/r9na7FrvWiJqRkMQoOewgYmG6FRlmoIAOqIpGuRlG9itJCYv25LTZLBdoiXbrNrW8hGwe5zosuVb3kALqUX1J24bB6B2sxvNh0CtgoKCOI779ttv7fRgc0yMmteDXW17bm1OPabWeatx1hI4dp5Sw7UBfLl8dHT0K6+8gu1cf/zxx1mzZrVu3ZoxtmTJkjotR6Zcg12atgYiYo3etBNr+qXcAYA2HLdMmuX2U6Y8AQtpboEcVPJeaI7ouq2Wq23WtCBJZvgADXEAQNodMYDYv6BYitwARQn1xnUQkD+KDUlfR516u8kINO83uiF9Kpdpsvk3SYHcZnn6JlWvVnOHIiB/luTpO6a58kbVk75tHAAiSfIY9Kb0VkREBGNMp9PVpqQUtUIjr7URbErVN6JsbQvrz2mEDURGCZZGKAHlJdugISwsbNy4cVh0y3Hc22+/jeXIjdAP21CQNixqnJ56SqEVcgeA4uLQtKKiorCwsKCgoODfvy7WvEJDQwNlr9DQ0LCwsMuXL+M5J8tRUVMAr6cVDbsFgv6bLLFxkfJ+lyUn9I1Ok2qrhhrHw94SElITKgK3HQL4objtzFYNVhFQEWj5CBDFqj9x2zgA4EAUqNPoDqDIipKSkri4OKxepUziW6RfCT4SaCEJJTuV8h01G8jQKtjaQDVQIaCmEW4qVVRUdOXKldjY2MuXL6emplIUFgk0IiE/e7gRxespguYfPXqUQoBooQVuHTx4sHp3Kawdh2NT+5NOhsYtnuc//vhjLGghX4twRj/WY9KNugVyL9P+O7aPfHmWaJsxwKc1duja7AHFESkkrJMftd5mW3GzKP32tsnIrFGTKgK3JwK37Bt9e8KlWq0ioCLQcASUiJ9d/u3kANBQPXaboWWmDQdFLlnnQl6qQj6QLC/VYtN2/Xrdy8Y1BHy00QswEOJCpLbOyBxaf0xLjR01lYxEwTprcVSnnTyqwDl0WAOAJwfICIJQVFQUERFx6dKlyJpXlOwVHR1dvV9TVFTU5cuXq3eyunTpUlhY2PDhw99//30shkb3yR9CyrGz5IZfNsQBgBHkBkiCNXbfSujp0yEzMepPpN8kSr+9a9Q6pFAVVhFocQjcsm90i0NCNUhFQEWgmRG4LgOEwG3jADR61akdrqCVIIXENUmGDjJrNPskVS0kofQcOGoe7cZD9L3R/8CwhShFucg9MQz8y5fGOmqn/JgCuTvXCD31FKntAKBe5GPfqvqfWIKRNq6dOXOms7Oz/Cji29IB+D1q5BQoJa6J/xZOhAyK/CE3wGxzAzCf8Ps61CsVgdsQgUb/ft6GbVVNVhFQEbipCCgRP7v828YBkI861w4gaTi0RLzsWKbBYIASWkJ64+hjw61tuqRdf9Olo5opxEXOSh1VghkAlKJFHcAZ3UHj6I0OASLXDrVQdzfC1HqKQK18BoCE4d7QpRLgcD4BArRNmzata9euOB4OIKMsVMnTpPwmJezZubVaissXr430ky21I3nkObXmBOAZCL/f7YeUXUvYabC/rV6rCNx2CNzKb/RtB5ZqsIqAioAjCBDxqD9x2zgAODJWkqSKiorg4OCIiAg6H8ARWCRQzMOHD2/btg0n/ubl5f3yyy/Z2dl0eBYU3iDu6JC1TRdW6v5GaI6Li/Pz88OBA02JvwKw8k84BnLAGx23Y7FYqqqqTp48efbsWVqx0IjG1lOEIK3TAaDwfcwAkLCdQnJ1KH/mzJldu3YtKioiZAgQJSVU9gYmQNBrVaDsANgoPkKAfhcIJCfxSNuUXmdqoEbG5nPUskLNUBG4XRHAl/p2tV61W0VARaAFI0Ccof6E4kFgaNqNOwcAISXyEVACUz7YXHvMOCwsjDHWq1evkpISbNdDPIkGp2vTR/kEgiRJzz77rFarjYuLEwTh1KlTjLEJEyaAk1HkBhUh/TRoTfEw8v2C6C6ZQS1Cgga5UZzUkpjJZJJnEn+laBm5JNJoKY2Xy4ef0RyIQS3lCIJQWVkZGhp68eJFebAT4YaDjQE+lfLw8GCMnT59GjlUKdqLTLn9VHXtTFQkh5G0QRhneNGzSw2v3SlUCxmQkZGh0WgYY7SgltpFepqSIKvqdAAaqNkOE4vFMnHixI8++qikpAS3qBbAi8sGKm9OsTocACt9F0WLJAlmwSJIoihJZoso1BwCYBEFi8WaafUSRZuEdM3xJsN+e6RlY/82lTUrgWlmAM6CKEkmyWKqY6tQeiRIuZpQEVARUBFQEVARUBEgIlFn4pY5AKA1tdk2dRiRNvAhiquOi4vTaDSurq6FhYXg3HIlxMKxLSOUyKN9gMLbb7/NGEtJSREEwd/fX6PRLFq06DdSIknEUIni12ZgZCFspuJyoKk4joWSS4L+Go1GOrAWVIYIt9z5IXZrR8oJLjmhRJqKoGrETZFYQkICWDL2QbIrgsB0eQPNZvPs2bMZY8HBwThUGFXjMGCSVGJjJttL3hHyZtZuBTlL1KEkD5zpEoDAAHgRqampbdq0Yew6zzZV2ogE4PojOwAWi8loNBjNJjgAFtt+PRZbII/VJ6hJmG2egFATRESPnxVzQaz5ylyj9Thlwrrtp638b/6AWZKs7kZNmJCtw2p/WRrRj2oRFQEVARUBFQEVgTsVATkdrZ2+Dkm6cTMAcriJPoIQiKJoNBoFQZDH5BCTliQpPT29rKwMo9QgiJBE80gSaqnN5HJIktS5c2fGWHx8PGosLy+HgwFtRDrJSLKQcoiAUoKYKLXCbq0CHQ0LSVr9Scrly2GpInmCmkYacJdcILoknWi+XAlIWFJS0l/+8pf777+/qKio9hgzGkVsHgmTyVRRUUGaofN3lK6mmqqqqtqHKMsL0knAKAHEKisr5TJoo3yWgLqSOppk5CFJ2dnZr7/+OmNMvq6jxrTm+as6AIJgXY8Loo/dOg1msUqwEvXf3jWeABwAUZQEs0UwWyTR9raYLWajRcQyX8EsmBBWZLGYJEnAIy2YLRaT2apRtDoA1jQ5E7aerP14N08Hq1pUBFQEVARUBFQEbnME7FiT3eWtdADAMs1mc2JiYmho6NWrV8GrBEGIi4v79ddf09PTQezMZnNqampERERpaWl5eXlkZGRwcDCKp6amRkZGlpaWxsfH79q1y9fX98yZM5mZmcSVIZaQkHDo0KH169efPHkyNzf3nXfeYYwlJCRIkpSRkRESEpKYmAg2efHixfDw8Kqqqvj4+JMnT+7Zsyc4OJhmG/AwVFRUhISEHDhwYPfu3VeuXCktLY2NjY2PjydOj0pxWVhY6O/vn5iYmJeXFxAQ4Ofnd+7cuYKCAkmS8vLyLly48Msvv5w9ezYzM1NOxAVBiI2NPX36tJ+f39GjR+Pi4kCpQZGTkpJCQ0PLyspQEfylMNsLOYWFhZcuXcrLyyssLLxw4cKOHTv8/f2Tk5OxWU1RUdG+ffsefPDBVq1aHT9+PDY2tqqqymKxVFZWRkdH79ix44cffti6dWtkZGRJSQmsMhqNCQkJ4eHhpaWloiiWlpZevnw5JiZGkqTQ0NBdu3b98ssv0dHR5IqQ8wDEzGZzZmZmeHh4QUFBZmbmiRMn4uLi4HSVlpaGhYX9/PPPK1eu3LlzZ0xMDJTgYRBFsbCwMDQ09Oeff163bt3u3bvDw8MLCwvpWymKYmVlZUxMjJ/tFRYWlp6e/u677zJmXd8C40m4uRJ/IAfAHrJrofwIATKZrXzcIklBlyL7Dxk+d/7SjX57j5wKDAqPC4tJTkjJTcsuuFpYUWW+Nidg/WPtkpq3raxZkqpEi1ESTKJgFCzwH7CTqFWWJhRsMwOYGiKj1HkAgkJNqAioCKgIqAioCMgRsGP8dpe3zAGQj46vXLmS4zgfHx/YXVBQ4ObmxhhbtmwZ6GxeXl7v3r15nk9KSoqNjWWMde/eHUPXX331lUajWbNmDR2rxPN8jx490tLSBEEAxdTr9ThrSaPRcBw3YsSINm3a8DyfkpIiSdLx48c5jvPy8pIkqaysjDHWrl27tWvXarVaxpiTkxNjbPLkybm5uWTe8uXLkY+TnhYvXqzVaidMmFBeXg6CDrMxk3DixAnGmIeHx7hx4ziOgw0TJ048ePDgF198wRhDTvfu3ZOSklCFKIo4SYoaxXHcwoULaYB84MCBjLHw8HCSLy0thR6DwSCK4s6dO6uD4KdOnTpq1Cg6gqp169bBwcGiKK5du5YytVrt559/Xl5eXlxcvGTJEthz1113McaqEZg9e3ZZWRnqnThxIsdxAQEBgiBcuXKF5/l//etfPj4+jDE68Wrt2rUYzkf/khtmsVjmz5/PGNPr9QMHDtRoNP379y8sLDQYDDNmzIAxAIcxtmvXrtLSUmCYnp4+fvx4VIEGchy3YMGCyspKtL24uHjZsmUwG3rmzp3bqVMnxhjBBb4O+Wb5JAeADgJzVK2dSS13DYB9w+AAWIf/BcFsi/63OgCbtu1i3F2Mv5dp72dODzH+ftbqb63bdHR2HTJ26rwFK7x1P+3euv3AwcOnAy9GJiVlFRVXWGzTBVWSVCZKBsn6Lhcko2R9I+Sn0mS21MT829wA27yBJGFSSB7SZgemvcnqtYqAioCKgIqAisAfDwE7xm93ecscAOoIURT379+v0Wi8vLwwyJ2cnAwmN3nyZBDB5ORk0HGTyZSQkNChQ4e+ffvm5eVVbwo0adIkCDs7O2/atGn79u3PPvts9ejvzJkzUcXBgwcZYzzPf/fdd1FRUWvXrgWrvuuuu0C4jx8/zvP8ggULRFEsLy8Hd2SMTZw48eLFi4cPH37xxRcZYwcPHsTI9JYtW3ie79Chw/79+0NCQrA0FvIY16dwFHCU06dPU41+fn7bt2/H/ANne+n1+h07dnzwwQdarXb9+vVyn4ExptPpYmJidu/e/eKLLzo5OS1duhRcZ+jQoYyx0NBQjOhj8B4sHHHwmzdv1tpe7du337dv35EjR3r16sXz/OjRowsLC7Ozsw8cOPDSSy/99a9/DQgISE9PN5vNO3fuBMhnz57Nz8+PiYm59957GWO7d+/GOLqXl5eTk9OFCxckSYqPjwfsjLF169bFxcWtXLkSObGxsdS58gH4xYsXw6n4+9//7u7ufuTIEVEUR4wYUe0V/POf/zxx4kRkZCR5JlhqXF5evmjRIsbYiBEj0tPTy8vLw8PDP/jgA8bY0aNHAYWPj8+f/vSnf/zjHzt37gwODp4yZQrP8xqNhud5ee1yk5qeRtXHjh37IzkA18b+cTovDuSyCNYzusyS5Lt5J+P+NPnr5TO/3z56weaBXj98PuqbLi7jnuvicu9THdld/2TaR5j2H+zuR+95+MVnXv1354/6OPcZ1WvwhKnfrlrq7bf1l9MXLqemFxsLq6xuQIVFqqpxAzBbgC6DT0g/YU3vR1WDioCKgIqAioCKwJ2KAP27rDPhsANQp5bGjcDR8HB8fHynTp26dOkSFxcniuLhw4exPvXFF1+8cuUKgkwYY6NGjTKbzTExMa+88srQoUPz8/MFQRg+fHj1mPFLL71EgfspKSkcx/E8LwhCcXHxxIkTGWM7duyA5ZIknT17FlQ1OTnZYrGcOHGC47gVK1ZgBqBt27YYqCbz9u7dq9Vqp0+fLopiQkLC/fffT84DgnzWrFkDBwCTEtixlOw5efIkPJC0tDQ8ZIGBgXAJoqKikBMUFMQYGzJkSH5+fmVl5SuvvMIYO3HiBEbTJUlKTExs06bN//7v/yLMZtiwYRzHBQYG0lNbWlqKRmErIT8/P8aYq6trfn4+XIKsrKw33nhDo9FERkYKgpCRkdGhQ4cnn3wSxypXr+v18vLSaDQHDx6ETrPZHBoayhgbNGgQcubMmaPVas+cOSNJUkpKCqoLCQkBz66qqoIPsHnzZvkJaxQLtGjRIo7jPvnkE0ReSZJ08uRJjUbTuXNnQkaSJKysffvttysrK/Py8oYNG/avf/0Ly7Vhxs6dO3me1+v1JpOpsrISSEIn6vL29tZqtXgA4CDBwjofXQKwEYk/ggMg49809m+N+beFAFk3/7Ee0itJG37axbh7J3+3for+zAjvwDH6kIk/hU3YGDplc6jXlrCpG4O+XHVk0MJdbjM2OI//4T2PhZ36zHzh4y+f7tJP+0hbdvcTjH/Y6h7wDzH+gf/38POfugyeMW+579bdJwOCQ8MvX76SmJScmpGZXVBQUFFRQRtGyR28Oju3cb9LjXgS1CIqAioCKgIqAioCLQ0Bpf+MyL9lDgDxQoyXDxgwgOd5jC7PnDmzdevWY8eO5Xk+ICBAkqTVq1dzHHf8+HFJkqKiop5//vnhw4cjEHzYsGEYKUdwsNlszsnJ+dOf/oRNYGJjYzt27Pjuu+9SdA3Yw5NPPtmqVaukpCRBEM6cOcMYW7JkiSiKBoOhffv2HMeBUGK7m4SEBI7j+vfvb7FYAgMDNRpN37590c0IMomPj3/uuee8vLzoGFfwTgxYHjt2jDE2fPhwBBGZzeaEhATG2KefflpcXIxdOPPz8xHXlJ+ff/nyZYQeUXwzwtxnzZrFGDtw4IAkSYMGDarWAPINSyorKzEDgFmUTZs2ITKK7BRFsVevXlqtNjo6WpKkzMzMtm3b/vWvfy0pKcE2RJmZmXBIUF1ubi7mOvr164c+ggFnz54VRTElJQVOGvSDaf3yyy+MsVWrViET/IzicBYuXOjk5OTn54e9Si0Wi16v5zhu3bp1mMHAqu6KigpMRBQVFWHhwZUrV7B+2mQypaWlrVixgud5X19fSZKCg4N5nnd2dkZEEJyu3NzcZ555RqPRYKkxnDSlrwGZ2ojEH9gBEGxLda2YCaJkEqQN2/Yw7s+ec9d7+gQP0scO9o0bujF+sG/cQO+owbrLIzbGjfC9PMo3avTGqLGbL0/46fL4zdGe+rDR686PXLL/y4W7Rny7rf907x5jlnw0eE7bj4Y98Ny/mdPjTGPzCjQPMc0D9z76f50/dh0ydvrcxatXrtX/tGPf/iMnT58PiYhJzLpaVGkUsRBZkB1PIIrWYCFabiBPNKKvlYrIHCQlkebPp5UU1q/Y79/NX5mqUUVARUBFQEXg9kRAiflc3wEQRbGsrGzs2LEeHh40lmanrimYgB+DJi5dupTjOD8/P0mSMBYeGhrK8/zatWslSXruuecYY1lZWZIkhYeHP/PMM19++SU4n4eHR6tWreAnIOL/6tWrbdq0cXJykiQpJCSEMebu7l5WVkZNkCQJw/xJSUmiKMIBWLhwoSRJBoPhtdde+8c//nHlyhWYZ7FYEhMTsXKgoqLCz8+P47jvvvtO3vCioiJnZ+eJEydigaz8Fg1pwz3AcDtinIYMGZKXl4ewoqtXr2LAvrCw8PDhw9W+0Icffkhb3MCSVatWcRy3d+/e6iXRWAMQFhZGi25LS1fZO9gAACAASURBVEuxLAEk+KeffmKMbdiwAWXR9r59+zLGIiIiql2ItLS0Dh06PPjgg/KtliIjIydOnOji4vL++++3b98ekfcDBw6ETgTrBwQEWCwW+DDwsugkgQMHDmg0muXLl9shgLmUpUuXMsaOHDlCd1euXKnRaPbu3YscxHZXVlZ+9tlnHMdhabIkSdWrmUeOHNm7d+/33nuvffv29913H8dxGzZskCRp8+bN1c7SvHnz0PXQkJ2d3b17d41GYzQayQewe27pkoxpeAJgiqLo7+/PGPP398czTDrlT1oD1QqCMGXKFGdn55Z2DoCNUdsaYWWdFAVky6nZl9MsSes3b2f8Ax5f67/UhQ/UJ7jrEtx1Sf19U/r7prnrU/vZ0gN8kwbpkwb5Jlo/9Un99YmDdFc8dJdH+kSM1IWN8gkb5RPq6RPi6RMy2id4nO7CyNUnBi7a7/r1tk8mrHut35y/dxns9H8fsj//L7v7Uab5O9M+xO574tF/vfnmh66fug0dMnrq5Fnfrlij27nv4IWQ8NTMnNIyg6HKZF1hLEhVFsko26HIunBZsLouZrNgNl/bq8i2Q5F1B1LBYn2LgvXT+hIli0W0WMC6bRmidQNTa3dbFzFbxQXJYhZM2L8IKNl2N7L6IyZTlTUfGx9J1rps+yDZPCcbecfTgrm+a7vf1pyoUEPuya+x0n1BEs22sxeubboq33bJah36yGa57aNGye8Ob6bM3+TUlIqAioCKgIrAnYWAnJbUTtc3A3BdB6CJQIGugaHGxsbyPD9w4MDc3FzG2LZt2wRBACfOzs4GiceQf0hIyD//+c9Ro0bhHN8hQ4ZQNDzYcFFREbaBr6qqwqlhQ4cOpf+ySEAAuwzBAVi5cqXZbC4vL2/fvv0jjzwSFxdHrbty5QrHcUOHDjUYDGCcS5YskSQJpFOSpOTk5M6dO0+aNAm7ZIJ00vj98ePHGWPffvttaWkpBr/hAAwaNKi0tBTcsaSkRKvV9unTp9p4rN/FhANmEjCo/8MPP3Act3//fkEQ+vTpw/M8QoAAo9FoRIR9RUVF9UlnGLzfvHkzWoGYnL59+3IchxmA5OTkDh06PPLII5gSEQQBMxU4Y+Hrr79et27d8uXLGWO9e/cGaFOmTGGMnT17VhCEpKQkirOHA2A0Gg8cOMAY++GHHwg6eWL58uUcx/3yyy+U+d1331Uvud6/fz9mCfAklJWVOTs7V5/1dvny5YqKiu+//x5+yOeff/7NN9/4+PhMmjSJ47gtW7YIguDt7c0YW7x4MTwrgJmfn+/s7Fy9qgR+CxpY+9FHDhnT8ATQqMcBaLgqkmzZDkDNdp5Wc69t/Qkqir35BdHmAGgeHPr1xpE+Ef31if10KX30KX03ZPXdkNXbN8tVl4F0P9+Mfr4ZfXwz+/hm9vbN6uOb6a5P7a9PtL0T+usTBuqvDNRfsc4h6C4P9I4a4hM1fMPlMVsSxm6L9/wp1tP30pi1p8Z+/6vHgl0u07w/GrX43QGzOvUc3955xKOvdm316CvsrkeZ5iHrKmTNg9b3XQ+9/XGPL6fNWa3ftvvw6bMhl8PiUuLT8jJyi/KKKiqqrAzfyutBhyWrP2Cl/pRj4/zXjjSzdZWNoMMtwPJkwSKazZIFEw+CJBrNtg1wbSjZfAAr7aZScm8cXB8PAJ4oehgosyb/d36XYPVNfjezQfuu2uy1dwBq1Aq2VRXXVMGvs8nX3Ff/qgioCKgIqAjcWQgoMR/k3zIHQB4mLkkSIljuu+8+BK+fPXtWkqR33333ySef3LRpE2Nsz549iOUICwt76KGHhg8fDvaMEKCwsDD0miiKmZmZr776KmPMbDYHBwczxvr160fBOTheAAuFsS3m+fPnOY5bvHgx1gC89NJLjz/+OCYHoDM1NZUxNnToUKPRiPUAU6ZMwS1QzNDQ0Orx5mnTphUWFmLUmeJeRFE8duyYVqtdsGABHfiVlJSEfYEQAiRJUn5+Psdx/fr1Ky0txaLkjh070nMIHrBkyRKe5xEC1Lt3b61We/HiRchYLJbS0lIKARIEYevWrRzHbdy4EdMIGIPv168fz/ORkZGSJKWmprZr1+7RRx/FSWSYeGGMnTlzJjc3t7y8XBCEsLAwjuO6desGYg0H4NSpU1iTgEXMOLwJ53zt3btXo9GsW7fu2olONuNoaJNmAGpojYQ1A3v27JGToYqKiqeffprn+dzc3OjoaMbYQw89dPDgwYSEBHTili1baHIDz8aUKVPkhyjn5eW99dZbjDGi/rC/zm8CgdzwBOxXHYBrY802LmmdAajlAFiJvswBsPH+9D6+6eQA9PbNctNn9L72TuutT+utS+6tS+7rk9hXF9/PJ97dO87dO6b/+pj+3tEDfC4P8oke6h01UhfzpT72S/1lT33UGF3EaO+Q0esChy49NmzJrwO/3dNv9jZXrw2fT1jzzoBvX3Ue++QbPdhfnrPOGNz9MNP+D9M8+D+t23/s4jFm6ncLVnhv2LZn1/7Dvx49GXAhJO5KcnZOfqXBaK7xCOAXXGPbVsptZc9mi8Fixk5F1ktwenIZzJJotog4/9giWp0EKLHtcGo92cwaqmQRzYJ1QsH6rnEvRMGM97W5AjoF7ZqcTRqI02SCbT7hWgUQszoeeNc8zjX51mkD62INhEpZN3GqkVD/qgioCKgIqAjcmQjUSXsos/EOQBPRAi+kCBZBELCfz+u2FyJwNm7cyBjDoU7nzp1DjZcuXXr00UeHDRuGXfCxjUx0dDT9K83JyWnXrh3HceCp//nPf5599lm5h5CQkIAocywMOH/+PGNsxYoVoigWFxe3a9fuoYcewqb+qBF8ffjw4TSl8NZbb2HjfPz737Fjh5OT06RJkwoKCoxGIxpFTcPI+pIlS8rLy8FEMQMwevRo7LIvSRI28Rw6dGhJSQn8jepNPKuqqsCMRVFMS0vr2bMnY+z8+fOiKLq7u3Mch/W4MDIoKAgOAOYlwJIxAwBkLBYLNlfFyoGMjIyXX375+eefh1uVk5PDGOvWrZs8Vmrfvn1OTk59+vRBFdOmTWOMwQGIiYnhOA5xVkajEQ7A7t27GWPr16+HPPUIWo3NOk+ePAnQBEHw9fXlef7bb79FNBe8psTEROzpWVlZifXTrq6uWIRgNBqrqqowCYM1AJGRkWg1poNQI7YohQdID7pSAqY69IlalBwAh1QRUJgB+PTTT1tkCJDDMwB9fZJ761N62xwAN32mq+7aqL+N/VsdADd9hps+08U3s5cui94uPum2d5qLT1qv9UmuPil9dGl99Cn9dCn9dEnu+uT+vinu+lRbTFFaP+/kvt4J/b3jB3hfGbA+xkN/Zbg+doTP5eHrI4avuzTSO9zTO2ysd8hk78AJPxwfu/LQ2BUHBs/Z9Jnnkn/3nfHcO+73PfMOu7e11TFw+hvTPsD+/HjrV955+xO37gNG9P9y4jivbxas8t6659Dp4Igrqdl5pRWVZslo2+/Ixp1tkwQIBLIFAAk2hDASjz1MQbwFwRoeZN0stYZ902h9lUX47YQD22oF2SNqsa6xhg8gdwCsaVvoEuYprDMBthxrwva2eQgoce05vHZhixuyOgDkA8j7tBHPrFpERUBFQEVARaClIyD7t1JHsjEOQLO0mPbYITp4/vx53vYaP3484lgiIyOxIX2PHj2Sk5MxoHvp0qW///3vo0ePxpY72Ocece0wLC8vr3379k5OTiCU2KJn0qRJCQkJJpMpOzsbnkb1ut7k5GRRFAMCAnDmAKJ6Xn311YcffjgxMZH4a0xMDGNs4MCBRqPx6tWrWIA7a9ashISE9PT0Q4cOIUZl4sSJCFICIaYNOrHXzapVqzCALYpiYmKiRqPx9PQkB6CwsJAxNmDAgKKiIpPJNGbMGGwrlJeXV1VVlZWV9f333/M8P2zYMEwj4PLzzz/PysoqKChISEi49957eZ7nOA5k+ueff2aMbd26FZgAZBcXF8YYVvqmpaV17NjxkUcewfaaV69exaLe8PDwHNsrJCQEG+y89957UDJt2jSO4/z9/QVBsFsEjHmPvXv3Ymwe8jQNAvqOGYCjR4+iH7EQGfR93759+fn5VVVVOTk5OMPrxx9/NJvNgYGBjLGOHTvGxcWVlpYC7bvuuovn+YULF+L4sy5dulQvRZg1a1ZBQUFpaWlycvJLL72EHqEerOPBr8mCqQ59KjkADimRCyNabMqUKXeGA9BPn2AdyLc5AK4bbCzfSvdtb980N9+0a2l9pos+u4fut3dPn2zbO7OnT2bvjXluG7NddRnwClx16fAZununfuGT2kNnVeu6wTqB4KZL7a1L7u+b1sc7oc/6+P76ROsiBO+4/utjhujjB6+P7b86cuja6GHrokbposdtih2rj5y0KWqib8gE3YUxP5wYtnRfv7lbP5uy5r0RC97qP6PdZ6Offdf9wTYfsn+0ZX96imketW5PZFuI/Gz7dweOnrJ4tc5vz68BQWFRcYmxCcmpKZkF+SVl5VXlVWaMroPi21yEa+P/14i6dG0dgsFspfYm0Xp+glEQqyw290C05hhtDJ14OoKNrI8KgnXA5q0TD/arBH4/D/Dbw4USotUxoLH/GhcG0UBqGNBvaKkpFQEVARWBOwqBGqZT91+HHYAbgQ1mA1JTU0EHf/zxR9SSnp6OWJ1p06aRnxAaGqrVagcNGgQHwMPDo3pzSQTDYK+YoqKif/7zn5gBEAQhKyurXbt2Go3mzTffXLx48fvvv88Yu//++6vXFWAGIDAwUKvVzp07V5Kk4uLiZ555huO4+Ph48DxaBOzs7IwheWyOCVPbtWun1WrBWadNmwYejG1AQX+NRuOvv/7KGFu0aBGaIElSUlISx3EeHh6FhYXoluLiYicnpy+++AI7BSUkJGCVQs+ePVeuXOni4gIujv1wRFGkWYIePXpMmDABO3LefffdHMcZDAZJknQ6HWNs06ZNIN/A09nZmed5OEvZ2dk9e/bUarVvvvnmjBkzDAYD/CKe5/v37//GG28gdOruu+9mjO3du1cQhMmTJ2MDfuwiiiUHFL0jiuKePXsYY2vWrKnzIVmxYgUtmUXolCRJe/bscXJy4nl+5MiRCxYseO2113BGGA55yMnJ6d+/P8dxbdu2nT59uqurq1ar7dChA9p74MABk8mEKC+NRuPs7DxlypSnn34adzmOIyezTnsanVmnA9BobZgeuTNmAIbM3TjcJxwOgKs+1XVDJhyAntfoe4aV/dscgF769F76jJ76rJ76nN/ePlZnoKf1M7P3xnyre+Cd0X19ek8fq6uAt+vGq919c77QZ39hdR4yv/BO7+GT5qrL6OWd5qZLd9Olu3gnu65P7qNL66dL6b0+udfalL76LHed9d3XJ7Xveuv6Y/d1lwf5xAxcFz5Mf/nLTXEjNkSN8o0Y81PUmI2XJvqGeK45M3LVCY+lh92/2e42fWPPyWudv1z6dp9p//fvfnc/2Ynd85T1TINWj7BW/2Dav3Xo4uzSf+Tk2QuX/LhB77f/5wP+B09cCLgYFROfkZ1XZjBZWb5g4/rW+B8bbzearMeaGU3X4vjlbgORdPIB6uDnDXYAatg/Lf+F+pqxf6zhbsqDq5ZVEVARUBFQEWjBCNRN/Gtyb6UDAIpM+4EWFxcvX76c5/mgoCDsv2kwGDZu3Pjuu+8ePnwYPAlRPX379p0/f35lZaUoit7e3u3ataOIHVEUS0pKvvrqq+7du6NTLBbL1atXv/76a/DC1157LTo6esuWLR9++GFGRoYoihEREe+88w42IDKZTJ6enl9++SV2HEJ4TF5e3siRI+mgYmyhM3ny5Ndee+3dd9/19/ePj4+vDtmfNWsWwm9ohBuXsbGxb7755q5du8B6RVEsLCz87LPP1q9fX1lZCeHy8vLu3bsvW7YMoSzwW7DVT/UC4latWo0dOzYxMZFaJAhCZGSkq6srGtWzZ8+srKy5c+e6uLhgV59ff/21Xbt22DgVjocoinPmzOnZsycO6hIE4ciRI926dbvrrrsmTZpkNpsLCgoWLFjwpz/9ief5F1988ejRo8XFxd7e3hzHzZ8/32w2b9q0ydXVFWuI09PTu3fv/uWXX0I5WhESEvLWW2/5+/tTn9L3QhTFzZs3d+rU6cKFCyiClc2SJAUEBHz22WdoiFarnT9/PkCg2KTBgwfj7uOPP37gwIHi4mIs/Pjpp5+wwiEsLAwasII5IyPjm2++8fT0pGeGXC+ypymJ5nIAoIccgMmTJ9/uMwB2DoCLb4bLxqxeGzJ76NNdfDOspN831fq2sn+bA+CbAVqPKCBXWziQ9dM6XWAd/sfbTZ+JOCIry9+Qe03YN8fVN8dVl+Wqy+rrm91bl9lHl9lXn9VHl9HbJ723T7q7LtN9Q04ffU7PdZluulw3Xa6rT45VsmbSwH2jderA1TvRfWPqwE0p/fTxfX2iB+suD9VFD9fHfLnB+vb0jfbUR3j6hI5cEzh6XaDnmoBR358csezY4PkHen615WPPVe0+G/vEG93Z315h2ieY5jHrjMFdj9/3+MttOn/6388Hugwe5zHO66u5i5f/uGH7viMXgiMys3IqKirwJaUHQJAF5Ft3L62J6KGpAxmVR7TQ7xYi1+z8c00LCcsecoz21ywPsEnYtkmViajJuhCo+UfZ0L916WiJeUrtaYm2qjapCKgINBYBpW868m+lA4AWgTsSqQJXk0ePgORhj07i0MTtoAS76yCN+QT654pMUFKMjmPbHMqnqmk7S9QOS0gPLsvLy4ODg8HFSYD2EbIzm4wklwBj0jT+DRvsPmlfTqyTLioqQkUQI9vIMLviNAsBATlixDXlWxjZFYcMLWMg8MkGPDc45JjKyuuistQp1CKSpwRBYTQaS0pK5ACiItRrNBorKyvRs1SWEkDVYDDADEIbAmQ5yTcxgVrs1gA0QifpwcN5xzgAffRJrj5JrvpUuQNwjfTLHICevhk9fWtCgyhGqCbRyzvV9k7r5X3tXbNCILPn+vRe3hm9fXN6++a4+GS6rM/orcvutS6117pUl3Wprt7pvb2zevtkuK3PdFuf2X1tqjWaSJfbwzvLTZ/d0yezpy69z+bcnro0a4jRhqyeunQX30wX34yeujRXfWrPtbG9dYn9dEn99An99YnuuoT+PleG+CZ5bEwarIsdvP7yEO9oD12Mhy7GthY5aqzu0vStEVN8L3p+f8Jz+ZHB87Z/MX7NxyMWdXKZ1OHzL1/uOuifbzjf9dir1iXIGusxZ4y/z/rW/vnDbi6TZ8zz2fDzsZOBYdHxiSnZKel52TkFBYVlFeUms6kmfOjaX1mMj/WLQQ7ANWYv9wHsJw2s1yQm12PbS6gRD+4fqQh+hRr+ebtgo9Si28V+1U4VARWBhiCg9E1HvqIDAHZSVlbm6ek5bNgweSRJQ2q9roySWdcteIMElOwBDlRpVFRU9Wlc7du3RyQ9gmGwPc7OnTvthKkUWHU9VdS+JS/bkHRtDfXnKOmUl5LLyPMbkpaXvRFpJRtuRF1ynQh5gqNy5MgRxtixY8doQ1i5ZMPTcACwBgB7W5HrSE5Cw7U1ryR2lblGG62qa7YBJUopWANasAsQZgDkDkCvDZk9fTO669JqRv0x9m/9tDkA6a76VDddHW9rBJE+1VWXTm9ooGkBWcI6CYC3m4814eaT4+ZT86m7VoWrPtlNJ9d5bXqhRr/1spc+A86A1XWBc2KLWeqtT3PTJbvpE930ib11ib118b118X1tb3dd/ABd/CDv+CHeMR7ro4evjxixPmzU+ovjNlwaqw/0XHfGY9WvQxbvcf9mW79Zm/tM9/lizLKPh8zt1H3c4699zu7/l23S4GF292OMf/BvT77y30/cRo2f/t2iH9b5bNm+65fDR06cOhsYFhWTkXO1wmi6tsjYtssQDfOLkmQ0myzWkwmswf3XDmiT37aebGCSRMFYVU6b/4iCZDLWLC1o3idG1aYioCKgIqAi0DIQUGJKyFd0AGB8SUnJhAkTRo4cict6iA5GZ2t/OgoCyFDD9dSWbFxOPXZiWS0ESktLFy5cyBh78cUXR40aNX78eCwq6Ny5M42114l4PfrrvKWEA4bea7exzkqbyyGp08L6M5XsqW05curX1vS7sKe2nobYI28LUXOz2Xzw4EEsiqit1qGcO9cByJQ5ALYQoGvBP79zAFz1yTXvVFvCRtP11mAhV91vDkMvPTRYP21kHZ/XiDvCh2hDIVsix/ppjTtK7uWb7KpPhHKbI2Hl+jUuhFWP7W3bj8g3u8cGq9n0tm5VtCHT5rHYjMEkhq/VSKhysW5qlNlbn4GFB+66pAG6+P4+sQN1MdZ9S3WRHvqIkfrIUfpwT32Y5/oQz7VBo9cFjvMOnLD23Jjvj49cvG/Q3C29JqzqOmj2K+8NeODZLuzPz1i3J+JtBxrc8+gzr777388Hdh84esjYmRNnL17248Ydvxw/FxKdnJWPvYlwmoGV89sN/ouS2fj/2bsO+CaOrD9ayZDc3XffXXohyaXdJaRyue/Sk0suIQRIARc1V2yDjQ2YjiFcIKRwtAQMxtjSrmQbCL0HCBhCJ0DAvRd1NwIGjIssaT/vPntYJK+QXMCQ3Z9+YvbpzZv33s6a9595M8Oca9DSbHWw8KB9p9E2OoPrhEvwgOABwQOCB25RD3CjF9fyNQDA5cuXY2NjY2JiOj0D4NokUDz0Nq7uIX+3s+EEEkjIgbSWlStXhoaG9u/f/7bbbvP391+8eDE+iKrbFQCB2A+uhc61yBf4dkh3bRRTOtd6V2pB095KwAo7VefSuTK5dK5PAFNBlhFs8Lp//35uxU6U+QAAVtUNkOtEc15V4Z0BaNsWnxlFvnoGIIczA8ACAMrCzgBAzM2M6LOhPLsCmLswANYGXFkhYPTXtK0TgGQhZsUwkzVk9Gd52r8BS1zNySYX+VMWX41puNY0XGscrjX6tn8YIczi4/btRxkkwE4gqJkVBWygzy5UaM9WAhCCv9tnDIxyNYs9SEaUP2Xx01YGaCoBNjATCGqDTG1gdihSMysNFGSFktIpqfKwVEOYtiKEYjKLwsniSHV+pCo3WpU1hsqeqM2Zlp4zc1XuZ6lnJiYdHjl/Z9AX632naD6MXfp60Jxnhoy//2U58dDr6PYnmXkDn37I5z72vLP/HfDyvyOixier07fv3HfqTH5+ka5MV1lhqD57vulyE81CACbUb2GOL7DbbU3MmQC2ZvYwZGESwKu3QWAWPCB4QPDAzeQBbiTjWr4GALBarePHjx8zZgzEQJDe7SrlOgQoHTZ6HdrFSepOz7ympiYvL+/06dNVVVU4UZ6P2aluT9zeKP90ly3u9Xf9la9dV06g4BMVnBj45PDRIfkHDnvetWsXQRBwMAIfvyd0JwAAvRrrieccPBHV7TyeAAB8EjCbAsQAALlKx64BaAcAaib6ZwfdnQEAG9AzsT7nw6wN4H58NRb4YAyAx+OdQYLG5MvWZb5ZADAs1QQfFgkwvwKDP4M9TAHtuUPtSUQmNvWooi0BiU1Dkl9JQ2qbNJCrTWyKEQMbAtoAQJWvxjJcU+WbWumnrfbTMtsZwSyElKqUaWqk2qqAtNqAtOoAZmcki59a75dcISeNStIQqK4I0+pHUOXhVFkEWTaSLBmlLhpFFkRRhTFU0bi00ri00glpJXFUTvTSYxGL9oXP3R48e410csonsQsGhc164b2wh/8+BPV9mD3TACDBnUhy90v/GjZizOez/pucqFm/cedPW3dmHPn5lF6vZ7KAHC20w0pbG2l2WqDb+4wgUPCA4AHBA4IHeoMHcCDRYYEXAEDE39jYOG7cuLi4OKvVirdt6dCqDqW7CdA94efydNgoxEZcth4qQ+t4NgAifnAR/MRdWMynaod0PoW5zHw8XDp3lJpb5vJwy1z5XSlzZXLLfDK5PNwyHz8fnVuXW+bj56N3oi5gAJvNdvDgQYTQ/v37uR2DryE3dC4A4DsIzE31Hv2piwBAqjEzgTgfAGBjdAjKcVwOA+pXf1/ZKhSSfCDy5n4DHcLx9u8qziSAqQ1CsHMI7csJmOQfNppnv0kjM1rf9mH2EVKomX1FmfXEbZ+2lQYMWlBXMR9mvYGlLX0IdjVlgQcDdTRV7OZFlQAz/CjjcAp2QzIrNBb/FL1fsk5OmZUas1SlZ1vRB1LGII0pmNIHqpmzkEM0urBUQ4iqbKTWODJVH0mVR2vLo6mSGKporLZwDJU9jsqcoMkas+L4BPWJuBVHxiw/MHLhduXn1JCx8/7uF/fAy/7oTy+g3/0NiR9CxL2oz92o758HDfk0KzuXWS3gYI8csDXiVQE92osE4YIHBA8IHhA8cP09wI1wXMvuAABN0w0NDePHj4dNGAmC8PHxgfJv6hv24EcIwUlbcIqwWCyGowDg2FrsEJGXF67YxQJfs10Ui6t7K99bftzQjS1gtfnUwA8dDkIWiUSwR21XXuybFQBAFhALEVQr1yPxnZFfpkWpc4JIZgaAWdrLnvXrCgBguW3b2DlnSB6C/vbwvT1F58pRwVUBJBt2Q/DdHoJDII4BgHOBujKZwPzUtpAAry3GKwFMcIwA830l4sehf3uaEGfGAGYPpGq8hKBtlgPmOmDrUoXGotBYlNpKeWqlIq1KkVYlI01AUWorA1OrFBoL7G4EnFBLTjFLDtilBaZATaVcZVAk64NIg1KlkyeXKpPLmGUGauawszA18wlNyQtJzh6hzovS5o9Nz45beWac5uQkzS9TqJNTVhyalrh35Ez1qPjF7w0bgUR/2PPTUfYkMOYsMmvLZQEAdOXNFeoKHhA8IHigN3vANejnUngBAAyuW63WnTt3zp8/f8mSJcuXL09ISPiO51rCcy3luZbxXAmciytyMc/F5fGkzBF/VTGR51q8eHFCQsLy5ctB3+XLlycmJi5dunTx4sVLlixZuHDh0qVLWuqNyAAAIABJREFUExMTk5KSvvvuu6skXn3DIz6Rxw3LuLWvaVdCQgKPmxndOrz42vVWTofCExMTufpzy3y28Dzeq8jcrsenP5fnmmV4iFz1cJkrn+uTZcuWffvtt999993ixYu//fbbb775pri4mC+/yM0MGM25bm0AEEBa/FTtyTNt6TRt6TcQkbdl47A5QmyaECQLXQnN8SLd9kQdzkg8s34XxuBd+ZkoH6J5VsKVSJ1BIIAr2Ea5sw1MJo+2LePIV9OW388sGGhXDzQElZizzEiDH2XwZ7c5al8ewGiC+SGmV1IW+ASlVbFHIjAMbYcZpzKzBL4qvZ/aAEeeMecbaC1SiqHLqcoAlUGprVRoLOykgSlkZY1UbVBojcyRamSFnKxQavRyqixAXSKnSpSakmBNcYimOCg5J1pTFJ2cHaf6ZUrKkc/V+yM+W4ok9/xw6NRlG3M8WfvxwG2nB3D6I1N08FxObL32lkd93lXP3vL3WsNvMcWE53JzPdDf2vPq/fbyaQh0dwAAVjq2tLTYbLZG9rJarc3ddDXxXNb2q5va6aoYm83W3NzcelAuHN/T0NBQX1/f2NhotVrr6+vh7OGGhoaWlhZwV7v6zv/y6cHjhiZcn6+iE51PjhMbvsXynQqYgVvoxHN3EotvuWI9KeOKTjp4a6/7trituJa5dZuami5fvtzc3Gy1WhsbG+vr6+G5871mnvz57hAAYFAB0ZgncnqCx10KkNMMgOSuDmcArgkAcKzcHug7h/Kw3Ba+r8YAXE48qH9VoW04nwUebDY/VGmbXmhbacBkIjFnEjNnDGuqhmmZbH5nGMAuLQBV2wN9Zv2xH9UGAJhjBNrhjYw0+ZNG/GFmCVRGuYr9pkx+ar0/uxLaT20IoExyFgAAHsDHojHnFVCWAE2lIrWGQQIkgwf8SIYoT69SrqqBE9Z8Sb0vqVeuqvYl9cNUFfL0Sl9SL08zBWj0Coo5CzloRWF4Su7IpJ+naE5IpyQgyb0/HDrVTNMtLABgMUDHvaYr/bljideX6q3+3vJfX2t+u60Jz+Xmeva/tefV++3l0xDoCP65uTpZV7Tlc0dXZPZE3d/ac+Hz4c3yvPj056M72WWz2fBBYDjix0sLnJi5t3zyu4veEQBgjgJgtpWx07SN+d3uoNWrNiDxneFfaKPUOYHqMiaTnjLIKebULV81sx736tC5bQYA0zEG6LAAA+rcYfUO2TokYlARwO772SEPM5CvYeYE2qYCYJsgzkJkvlqe07lquK/FHkRgdvp2reLE0H7LHF9w5RADysTCgPIwMm8MlSmbrkaSO3868jO7YagDzg3wtp9A3/O21vXn574j17Pc05Z20ZZrqscn/5oVe4iht+nTXWby2cVH7652b5QcPruA7qoVH92V8+ai8Pmhp63ga1cAAG2e6ekH4K38W/UF6JwfXLuvt3J6G7+TRbcSAGCOylLrmQQVHgAA4Wxbxj8ntcY1zP0NUtpD+aswgKsfOmSDZQNyJrmI/VCVDEVjCCSLYzS50hkUkvz50OGjLHgTAIDTK9g9tz39d6aLWl5TPT7516zYQwy9TZ/uMpPPLj56d7V7o+Tw2QV0V6346K6cNxeFzw89bQVfuwIAaPNMTz8Ab+Xfqi9A5/zg2n29ldPb+PHzhcI1AcCN0r8TMwACAHCN1z2ndBjZe15dziwbuAIAmHkPSi8AANc/IE6UG/V+eduuk9re3nrb3A3n5zPwhivWRQX47OKjd7G5G16dzy6gu6rHR3flFCieeIDP/wIAaPOMJ068njzCCwDe5uu41/NZ9ERb+PlCwW63u08B6gkdPJEpAADPg+9u4ewuAMBsdcrueuQeAHj7fuF+60nnuYE8fHbx0W+gql41zae/h3Sv2uoNzHx29QbduqIDn1189K60dT3r8unvnu6qIfC70gVK5zzA538BALR5pnNu7blawgsAvuXruD3n+esmGe8RBIuABQDQLQH0zS6kiwAAFiLDQQcCAOD76+FKv25vfRcbctXcK0oXW7/+1fmsu/6adG+LfHbx0bu39Z6Txqe/e7qrPsDvShconfMAn/8FANDmmc65tedqCS8A+Jav4/ac56+/ZKddgDpcBHz9tWrzP21n9oph1o0Cgbl1vwhYSAHqCgjpCQCgVBfxrQHw9v26Wf4ueWvXjXq/vG2Xzy4P6d42d8P5+ey64Yp1UQE+u/joXWzuulXn09893VU94HelC5TOeYDP/wIAaPNM59zac7W8fQH4HnDPaXh9JPPZxUe/Plp1byu/ZQDQYbwr1Zi7EkPf1HU7dIjnFjnNAPiTRimpuwUAgLfvu7f83ftG95w0Prs8pPecYj0kmc+uHmqu02K91ZOPn4/eacWuc0U+/d3TXZUEfld6d1H49Oku+b1NDp+9AgBo80zvfGCea8X3gD2X0Ds5+ezio/dOK9xrJQAA16jX85D3FuN0dYVXcIgLAJjDBAQAcPVfCvdvYu//9WprvL7r/QY6achnoRPbDb/1Vk8+fj76DTfQQwX49HdPdxUO/K707qLw6dNd8nubHD57eQEAJCjbbLbrbAmfot7SPVGbK5OPHydq8zH0EJ2rG7fcXc1xZXLL3srn1vWk3NPyPdGBy8OnD5cHypgTXgqnTfrxr50oOBwO2AVo6NChFy5cgC7ndBCYkw6daKVzVdwtAu7oILDRVJ6SKperdAHqChnJHHbrT1n8yLYR/fbwlDkHALYB7TDexcRbLLi/Dua0e5g5pMxpDYB8phaJ/3Tk6HEmh8tB22xWh8Pm2s9vVE/rXP8UaoEHhOco9ATBA4IHsAf4/iA40XkBQHNzM03TdvbCQm+lgpMjrnl7K9nulS3uPeOVqF7IzGcdn6oYGLe0tOBkfT5mD+m3CAAQ3xn5ZZoAAK5DlO+mCQEAePjS3WJs3v4du8XMF8wRPCB4gOsBvj8ITnReAADxDVcijnicRLi/5Urglt3X6vqvAF1cv7su2WlOwEkgttGV7kTp3K2TRViIEx3fYgYPC3z681XH/B4W+OR0L91DZdywcfXhssHTdzgcLS0tTj2By+ZVuUMAgBUAmAG3XontFmYvZgA6BQD4YlmYBOD7VaDzeaBbAEC39BxBiOABwQOCBwQP3BAP4PjBfYEXAMDwPwQfONvhhljSuUZxBOxU6Jy03lOLaw730XLp3HJPa87VwZNyd+nDF3y76tC5Frk+5MqkaRrnxQEGsLJX51qBWjcrALDTtJ3ZHcjuoFPS1yHxnRFzUr2dAeALZAUAwOcZ9/QuAoCudGOhruABwQOCBwQP9AYPcIMWN2V3AADMsNmYPFEo44KrhXxtuHL2Hgqfzm7orso7MWMGJ3qHtzjP2/VXLMepAJxOxJ6+7cRzd7WocxRvTfO2FW/lAwDAkNhqtdI07Tpd5pVYAQC4BrUCAHD1iScUjwAAu6mrzWalaTv3ffGq0wrMggcEDwgeEDzQOz3A/cPupswLAHBsCsENDnHwyKiT0N7pheupFTjEtUUu3clpbm5d5XSOgp+XU6Fz0lxruTGhw59cJXQvpcNG3RC72DpAIzwn0DlpfACAu8YATOic/K7UcpcC1D4DYLMLMwDMsube8OEDALHaPGYRsOTPzCJgFgDY7S0CAOjKqyHUFTwgeEDwQO/0gJuYh/sTLwCAcU273b5jx45vv/12wYIFi9lrafu1zLMr0curXXxX/+XTjqsOH8816YmJiR3yLF261A29qyax9V3lcy1yLXvbqKsEMLZDOp8fXJXElOU8F2ZwKniiv1MVr24TeC7o7YsXL17S0fXtt98mJCQsWrRo4cKFJSUldNeumxIAQPTPfDsEANAbQn/QodMAoGtdWKgteOCW8oDDy+uWMl4w5ub3gIf91x0AcDgcly5dGj9+PEEQr7322ssvv/z666+/y3O94eX1Js/1Vjddb/NcuFkv9b02++vs5coHdG+/sZ5OBVez/uX24nlc7/K52am5N998EyxypQOFr/F3eC6+dl395r5drj5cmU768GhxhfxvL68PP/xw0KBBAwcOHD58+KuvvioSiXbu3El37boZAQCTFcguAKDtDqvrGgB1ObsNqLFtG1DKeHV4emUbUL7QmT8FiN3gkhEIH1aUxuTPftpbMUEcLKP0Vz6kUUaa5GqTnDTKST37bZQzRPiYZKQnQ/isIaQlgLQw25hqrlSBXU2ByKW3G9jmgfbbKxW7l8IBACbuNqAdzQCw6M3BXF3rv0JtwQO3mgfY18KLr1vNfsGem9wDHvbdawOA2NhYuVxuNBrLy8t1Op2e5yotLS0rKzMYDKWlpcXFxeXl5SaTyWAw6HS68vZLp9NVsJfRaOQRc4UMnHq9vrS01BP+KzXZktFoBG0NBgP8VFpaistOzG5udTwXqOf6zcPeZrgrv3sKSNPr9Yb2S6/Xm0wmXEun0xkMhvLy8vbfPf3Xjck35CdskYcFPiU9tb+dT6/Xm81m6Ksgs6KiwmAwmHguo9FoNpstFsv58+d37dqFENq7dy/k6nQukHKwl81mmzx58tChQy9evEizF0jjfgO9h75BjQ6+mSW+dkgaYZtmbtsCRjaAZHaUp+lE7WokuStiTmqUOktJlSvICimpk6oN/qSRiYa1cA5AW7QNsTs3ZPc4CDYGaPQBmithfQDFyPfVmIZT5uGU2Y80S9VGmdogV+mUqtIQbXmQtlxBlgQk5clWlMhX6BVJemVKSQhVpkguDlbrAlPKlSqdkjQoKFNAst4v2SBVm6VqcwBpYRBIKvPxpyy+apOf2iBjD9WSqg1yba0/VfOpqtJXY/HXGGVpDBgYTlV+oq70S635VG30Y+0NSquSsrWGp1QEaPRQ5rOUH/B4BxVkJAfJsEBFRhoDyZJYbYHyP+mIuOPosZ+Zp9me/N9D3anTYjvogSyp0wJ7eUU+e/novdwcrN7Nrj82RCh06AHh+Xboll5I5HtSQOcFAC0tLTabrb6+ftKkSTExMcDtuXl4lSSsJcC315Rgs9ns7IU5OxdXQXXcLi5gsde54P4xuP6K12C4mu+0KgPv13SdLbr1mrOxlyd2QXc6dOiQSCTat28fBgCuD+ua0uDR34wAgJ0EYLAABgAjWAAQqCnrCAAwA/ac8WkmcO9opPya8S6DAWSUXk6yMIAFGL4ai6+myk9bHUBa5GpTkKoiVFUcqc4fqc4ZpTo9Ji17VMovYzQF41PLopPzYsmscWk541ILJqwsn5BeMVpTPoIqD9IapCo9c3IZG/3LNFWKtComstcY/SijH2lWaiuD06oUlCFApVdQNTLNWT+y0o8yytPNAVrDMJUuQFP9qZpBC4q0KrnGoKAMclKvoEwKjZkxPM0iT7WwyKdjA3sGAFQFkBYXAHCSBQC0t3/Sr9mTu4XB9S9h79SzW4zl/t3gM9yJ3l3t9rQcJ7XxbU+3K8i/Ph7AD9SpcH1aF1rx3ANOD8jplhcAQBReX18/fvz46Oho+FOFlwK7Ng+hD2yKAr86HA5u2A1lLsVViBOlqampExEVFgJtgQQ3mmP+XlVwek5YN+wQQAhA98qrWFQvKYClN1AZ7FK8nw8XfXX4IMDhBw8edJoB6IQtUAWfBHyjZgD4/M+7CJimOwQA0WQ2FwAwA95aMzsD0B0AgEm8gaCWSfJhMACTwGMJIKuGJpk+VVVKqWq5yhCaXOT75d43IxMHjvpu4Ki5AyO/GDZ20dDo+UMiv/Ydu2D4+IXD4hYOHb3ok9jEj8ekDB6XJvv2WORKQ2CqTq7RKdKZeN1PxYbpWpO/Ru+vMUo1bO6QyiBjR/QVGos/aQxQVyi0+sA0s7+qXE4xMw8BKbpgSh+p1YWrS0OoskBNhTS5XKo2KFJr/FOrfFMrh2uvyhrizgYIAICvBwp0wQOCBwQPCB7onAecAhinW14AABubNDQ0jB8/PioqCtrmhkodagO1WlpaXENS2E4UAnEoO6kCtzCezd1WBRrtkNkNERALMzxps3FVdVOlw5+4da9nuUNlgMjdmJWLuLpFPTftdstPoKSrKG+Vd5UAFG/l4I4KiBc62zUBALAdOnSo6wAAOuotBgCUlE5GMUkvGABINSYmjKaYD5TbgUHHI+Lc4PhKmQEANf5UDTuwDdn8RrnaIlNXBVBnP11h8U82+y4tDE3MfnVEIrr9/5DkKSR+GIkfROKHENEPSe5F4juR5G7U92EkeQJJ+qPbXkK3vfrRrJ0j0spC0nQyqkyq1bHpOkYmoFfpmZUMlD5Qa5Eur1AkGxSwhIAyK7RGOVUmI4sVlE6hZWc21OWBZElQUrbfFxkfz9zx0dcZoWSRPLk4OM0SoDX7plb6p9V+rDbwTXoIAMDbN1fgFzwgeEDwgOAB9x7gi5SAfg0A0NjYGBcXFxERYbVabTab+3F07tmoEFdBnOQUguPQvEO9gdlut+N8DFzokP+aRKyzqxrXrAuquvdgD/3K1Y3bBE3TjY2N8CsXCXD5b8Yy10ZPyt1oI045g64Cnda9fPD8gQMHEEI//vijK2BwX93pV+4i4FtjBgAAgIxkV75qzdJUS7cAACYdn6r1pWoDyCpmRS9pVKiNTFCutvipqj5ZwSTwyJYXj0rJeysqGf3Py8jnWdT3adTnSSR5HIkeRH3uRz73MN+3P4puew71/Qfq8ya6/T3ZgqOj11aGpRqCU40yslRGlspVOlmKXq7SBVPGILVJkWwIUVsUySZFsklJmoOoaqXGrNCUyqliGVUWoC4LXGWWUkWhqfmxqVnoCT/0h9d8/jEiNj0vXFMkTSkepioLSKv001YGaCqvgJmrtw0VAIDTSyHcCh4QPCB4QPBAFz3gPpq6BgBoamqaOHHiyJEjPY+e7XY7DEvX1tZevHgRx9/YDDzmiikdFiDGamlpOX/+fFNTU4c8boiQgISDuV9//bW2tvbcuXNuqnT4k3v39dyvTsrghpqamurr67m/cnOBuPSbq4wN9LDQXdYB1ARpuGdC33OjCbwOhw4dIghi//79fEDRQyVvMQAQpC3vaQDgT2EAwOznw4ABba2UqpaqjdKk4pFk0egVv0Qt2Ds95ejUpAOfpxz8InH38NCZCN0tur3f+HmaGao9U5MOTVzy80xV6VSyIiatPHq1ITJNF06VRWrLRmhKgsmSQFVxoKo0SFURrDYxn2RDYFJ5cLIuUK2XpRikakNganlweoVUXSRTFQVqysJSiyPI0xNTT6Hfv4RuexE9MnDyqqyRKVlhVLEyVSdNNQ9P0UmZ9QAdz3gIAMDDl0VgEzwgeEDwgOABDz3gJoxxOBy8AACkNzQ0TJw4MSoqCsdG7lsFtpaWljNnzkil0tbdErOzs6EKROTXnEYAZgiwbDbb6tWrfXx8Dh065L7dDn/FymRnZ4eFhSGERo0a1SFnLyeC6yBUXbduXXh4eGVlJZ4YwSCnl1vBpx50UL5fe5oOw//Nzc3QW2pray9dusQX0GNXQwEWAWdkZFyT370VGAB89NFHN/sMwGgqhwsAmLg21dKNMwDDNVXDNVX+lAVmAGApcABllKaaZVozszBAXa5MKQpekTNiRVZk8ulRK07EpRyfkXRwcNB/kOhedPvDEd+snJh2MkqdM1JVGpVcEbGiLFSdH6EtGJNaOGlVyThtbgyVFZWaG7WyJDy1UJFcIE8qCVMbw9VlI8mSaG1pRGoZk/+TXqlYWSFVF45IK4vQlEZry0erCsarT89ZlYnueBnd1h89/M7nq86MV50ZTRYGqYulqrIAdYWcJ/oPYNKizG7gAR9scKVfvQvQzbcI2P2bIvwqeEDwgOABwQOee6BLAODSpUsTJkyIjo6GzXncBzo4PLLZbMuWLUPstWLFChi/hygHR3t8auG0FofD0dzcrFarEUJ79uzh4+ejg8IQOr/77rsIIbFYPHv2bD5+Prrnju4hTi5wamlp8fX1FYlER48ehWeBgRmf/r2cDvE3tsJzH3aXXdymWzecDQ8Pnz9//jVXVkAtvkXAXN08sQge8ZQpU24ZABCo0cs17L6Z3QsAmB0/Lb5t2wexmfeUnt0JVO9H6fzZhQcytU6m1klVZf4pRTJ1YYg6P5rMnrji6Dvyz5HPI5K7Xwiduy42NStEU6xIKQtOKY/W6GI0uWPInxWz178a+OVryi+f9535f6Hzhs/ZOFp9YjSVE7wiL2h5/lvj174YvOT1UYmKBRkj0sqU2goFVRymyRtNnh40ZfXfld8OHb9y1NxtyslJ6H+eRj6PoLv+/mH0twNj1dFJmZHaMpmqJDDNKGvbDrWDSQABAHjypgg8ggcEDwgeEDzguQe40YhrGbkXBLsARUVF4eCejx/nCDkcjqKioueffx4hdPvttyOELl++DNEqty6EUBBp4fFX0A8z22w2kiQJgti1axcsDuaqYbfbuUnb3KAN2OC7srISITRgwACdTgcKcIVwy7hd191guA3R7MJisBdCN2wXzibHYSWWj/kxs1MBOwR7Esb4wUyQY7fbf/nll4SEhObmZszGTQHCDnQSDrdYK5yzzjWZ62FYxo3Tt2w2G87CAiG4ddwQthRqweOApSPYn8AMTsP8Tp0SqwQNYZ2xB3CLsEkUAEssubm5GTitVisUAFLa7Xa8dgIsBQ3xE3Q4HMeOHSMIIjo6GuvGbR0TsQI//fQTQoibAoR/umYBpOFvh8Mxffr0QYMG3UQzAIyNDmY3yRaaXkqtZM4B+Co9RpMbnFoBAIBd78tsAdRdMwABlMlfY/TXMEF/+2ECsJ2o0V/TtssQNCpNtfilVfmnWhRafYS2eCKV+e/QuUjyKHH3S+HztsSm5ynVRSEaXVhK8egVZ6au2P/CoJHscuGHkOhhJH4C9X0KSf76/KCY0Qt3xKScGrXi1NAJWiR5Ht323KNvRkQsORxFFUSqcqeszHlv5BeoTz8kefKT6HkvDopAfR9Bfe5Bv3sA+TyE+vwVPf5p2HdHR6WWKsgShdboq9K5jtl3L8XNDIB0ugaJ7zx67GTbCW7X7KMCg+ABwQOCBwQP3MwecIqvnG67DQBAUAUB044dOxBCE9kLIQRHpeI4FRcg+oHIsrm5OT8/PzMzMz8/H2I4CHwBAOzdu9fhcBiNxtzc3LKyMiwBnguYdO7cOTiMDFLkceim0+kQQgMHDszNzW1oaGiNtHBUCrMEVqu1uroaRFkslsLCQnwLRGCAzBCdTldYWMiNhs1mcxl7NTQ0QPAK+jQ0NJw7dw6wTWFhYXFxsdFoxAKx5lartaysLC8vLzc3t66uDuMEiP4vXboEXjIajZmZmfX19XV1dWazubm5GaLqixcvYvMLCwuLiopwBAlrssEP1dXVZWVlZ8+ehXYbGxtx+AsUp++mpqaCgoLc3Ny8vLxff/2V+yvow33cNE03NTXZ7XY4pAyccOHCBYzrwJampqbi4uK8vDydTof3iQL1Lly4cPbsWVAJPABrPxobG7m3UK6oqKisrKRp+ty5c01NTS0tLXV1dSUlJXq9Hs9TnTt3rqCggNsQKAzg5OzZs7m5uRUVFdgh0NkOHDjw4IMPhoaGFhQUnDt3DizFHcnVYwIAuO4AwNh2si+7vJg5SYCysHMCV7bXZBJptGb/1Co/ZvPNGpnWHK4tH09lvRM2D0n+Krr7n+Hzto1NywvRFMuS8iJX5IxdfuT/BkcjnweR5IHHX/jAP3yWcvR/Xx0UzawbJh559dO4CarDk7WnxyZkPD9oDOr7BJI8PnDkws9Sz0zX/BL51TokuR/d/mC/f3w6K2V36LSln46Y3ueOx5D47j889PK7IbP/HZscmZwZpi4KSTMo06v4tgDqRgwgAADuHyuhLHhA8IDggd+yB5wifqfbbgMA3C2AfH19EUKZmZk//vgjQRAjRoxwfQCgB4R0BQUFM2bM+Nvf/oYQeu655z7//POKigqapuvr61NSUgiCWLt2rVqt/uCDD+67775///vfc+bMoWkaAjIYj//hhx/CwsLefPPN119/fcqUKbt27YKR3dzc3IiICITQnXfe6evru3TpUghPcQBqs9kOHjwYGBhoNpu/++674cOH//Of/xwxYsTy5cuxzj///PPQoUOPHDmSkZExaNCgu+++GxYTnz9/PjU1VaFQDBgw4L333ps1a1ZOTg4eUVapVJGRkXv37l2xYsXbb7/91FNPtXJu2bIFD1HTNF1VVRUfHz9w4MCHHnro+eefj4yMTEpKwhH2vn37wsPDjx07lpGRMXz4cITQuXPn5s6d++KLL+bm5tI0nZmZGRMTc/z48RMnTowePfqVV1556623vvjii+LiYiyksbFx5cqVo0aNGjhwYHh4+Jo1azIyMqZMmXLs2DEwEAe4eLy/qKho+vTpr7zyCkLo6aefDgoK2rdvH/zKnY7ATdhstmPHjk2ZMmX48OEymUyr1R45cmT8+PGbNm0Cb9M0nZubO3369JdffvnRRx8dMmTIggUL9Ho9+Kq2tnbOnDlxcXEWiwWrlJmZGR0dvWHDBlg0kpCQMHr06IsXLyYmJg4ePDg+Pv7s2bOTJk1atGhRSUlJdHT0K6+8MnTo0G+++cZut+/duzcyMvLFF1987733vv76azwVABsorV69Ojo6+oUXXnj77bcnTpyYkZEBHpg7d66vr2+fPn3+53/+57XXXps5cyYX5HAz03DH2L9/P0KIexAY/umaBfAe9qEwA+BJECxjtv3RK9R6udoUwJ4J4MusB4CNQasCSCblHXJpZFqLQmNRUpZg0hxJlU8iM/8dOg+J+4vueiN83o5xaTmjV5ZFaUrGqfL8JmoR8Qgi7nvpvdDJCzZ+lrR/ZvLB+GV73/CdyGwcJO4X8vXqOM3RCdTxmG+3IclfmDH+/31q+pL1sxO3PPumlBnp7/tExJfr/pP2y2fqo3NTj/Tr/wHyeey+Z4f9hzoVS+aEa0sD1WWBqZXK9GpFeq0nZnaFRwAA13z1BAbBA4IHBA/8RjzgFPE73XYbAGhubob+9P79AAAgAElEQVQYsbS0VCQSEQTR3NxcWlqKEHr44YcLCwtxrAMhO3jf4XAUFhY+++yzCCGZTDZ16lS5XO7j4yMSiQwGA03TKpUK1hKIxeLw8PCxY8c+88wzCKFly5aBwEuXLn311VcEQQwYMCAiIkKhUPj4+BAEkZiYCAkzcrkcIXTHHXd8/PHHixcvxk8dsAdN02lpaSKR6K677hKJRHK5PCws7IEHHhCJROPHj4fQMCMjAyH04YcfIoSeeOKJ4ODgCxcuWCyWqKgohNCbb74ZExMTEhICep45cwai3ilTphAEIRKJHnrooXHjxsXFxT388MOgOUAXk8n00UcficViuVw+adKkuLi4f/zjHwihFStWgJJbtmwhCOK5557z8fH585//7Ovre+HChbi4OITQyZMn7XY75KD/+c9/FovFAwcOHDly5LPPPiuRSMaMGXPu3Dmr1VpfX79o0aJWb9xzzz2hoaH+/v4EQYCeKSkp4EAMAECr2tpakUiEEAoMDIyPj4+Li/Px8RGLxceOHcMewz6EwqZNmxBCBEGEhYUFBQWJ2EssFn/55ZfgigMHDohEIrFYPGbMmOnTp/v6+orF4v79++fl5TkcDrPZLJVKH3300fLychxz79u3jyCIefPmQb8KCgpqhZSffvopQujJJ59ctGiR0Wjs16+fRCIB+tixY8G9gwcPJggiICBg2rRp7733nlgsDgkJgS7ncDheeeUVkUj0+OOPx8bGhoaGIoQkEsmWLVtomk5JSXn//fcRQk899VRERIRGo2GSW1pauN3VaepJAADXeQZARpoUamOgitn7P4BkQn9fqsafrGVPBqjBAEBOmZQaYyBlDKaMoZRlFFk+kcpkZwCeRXe/ETZ/27i07JjU4liqcKoqs+/DHyDJw7+/97npizfHL8+YuPTAhKUHJifun750B5I8iMQPPD04esbqU2OpExPJ4/5jF6I+jyDi3jc+VChGTkE+96PbHldOXPH5mqJx2uz4lfn/XVd459+GIuKvf3om4LP0vFFUYViqQakxKjQWubZanlbTLSt93SAEbwEA98+y03st3AoeEDwgeEDwwE3tAaeI3+m22wAA+Mhms82ePbs14F6yZAlN02fPnp04cSJCaPXq1TBm73A4IG0DBsJbY8rPP/+8dZ3A5MmT6+rqmtlr8eLFffr0gWH+pUuXIoSGDh2KR6x3796NEAoLC4PUlH379olEosGDB5eVlUGstnnz5scee0wkEhmNxpaWlkuXLolEouHDh+M1APhxwgh0enq6RCJ5+OGH9+zZA6k1hw4dGjp0KELo4MGDNE3v3r27T58+BEEkJCSUlpaePXu2paVFq9WKxeJRo0bp9XpoNz09HSH02GOPQaAcHx8Pip08eRJazMrKgti6uLi4paVl27ZtCKEJEybU1dUBw/Hjxx9//PGYmBiIxbdu3QrBOkVRp0+fvnz5cl1dHfgzMzOTpunjx4+DwMTExNra2paWlsLCwscffxwhVFJS4nA4jh49ShDEQw89lJ2d7XA4Ll68SJLkn/70J4Ig0tLSOvy/f/HixSKRaPny5RgYbN26VSwWL168GHKcsPcguK+oqACcc/To0ebmZqvVumPHjqeeegohNHv2bJvNdv78eVAyPT0dZNbU1EydOpUgiPHjxzscDovFEhIS8sILL2AA4HA4Dh8+3Dpvk5CQ0MJeEKz7+Phs374dvFdaWvqHP/wBIZSSknL+/Hl4TIBDFixYcPHixebm5szMzD/+8Y+tWO7ChQs0TZMkCcAmOzvbbrc3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HA5HZWXlsGHDWjdWAgBgs9nq6upiY2NFIhF0lfj4eIRQYmIiKO9wOM6fPw8pWJCpBZsCKZVKmMPBNnIdxQ3cXRcB467l9Aq43nLlQK+7sQAAx/14xgN20d27b4/V1syOGUPmv72hqdHmsNttrE3scgAbTc9fRiLx3eFfrowmcwM1FUqNEQBA21A9MwPQMQCAqBfG7PFIP6wVdv2GGYCrhv/5ZwwYAECZglKNEdriCeSZQRELEPFYn7teHvHl9lFJmeEpueFUTrTqZ994NZL0Q5IH//qqPOI/aZMS98Yl7Bk9b8srn05g6H0fC561YSqVE5/8y1vSz5HoboT+NFg5dVbS3ilLfxwY/hWSPIyIR17x+2xKyvGxScena0/dOyCAmQHo+7dpS3aMXvZTlDonQlMRlspk//upDd0S5bsRIgAA13dNoHTdA57/Zet6W4KErnug68+r6xK6boWrhN6plauevYcCHuP77jYAcPnyZUhouXDhAoye4sjY4XBARsc777yD4wzsoD179ojFYpFIlJ6evnnz5qSkJISQWCwuLi622WzLly8nCGLv3r2Y32g0Dhs27P3336+oqGhpaSkqKgL+hISErVu3rl+/fvTo0QihmJgYmk2+r6mpaY2zQ0JCYPtODCSgYLfbKYqCmDgiImLTpk0bN26cMGECQRAvvfQSJBFB3s5///tfnBZis9l2794N21+mpaXt379/48aNkCsfFRUFNk6ePBmLXbdu3aZNm0aPHg1LYy9evGiz2ZYsWQJ6bt++fcuWLV9++SXw9+vXb+/evXa7HQDAd999h22/dOnSuHHjRCJReXm5zWbbvn07ZPMDA4RusPYAxrnPnj0bHBwMGVCtZs6aNQuagGWvoCdEolCur6+HffTj4uI2bty4efPmGTNmQIj83HPPwX750BaeAbh8+TJsKzRs2LDExMS5c+fC3AjsAgSbAo0ZM0YsFoeFha1fv37z5s2AuFqPVvj5559B2rfffgubvX7//febN2+eNGlS6zECEokkISEBHuKMGTMgXwinplRUVNx///2wBgCUt1gsgKngEN+WlpZz587BIuCysjKHwwFTGSKRaN68eRkZGdu3b4cVAlFRUbC84dSpU0899VT//v2XLVu2fft2GIkH4fDN7Tx8MwD4YXlegAcXHx//wQcfXLx4EdrFz4ULFTyX6S0ntMKdDqJp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PiYmZsuWLbGxsfn5+X6/3+127969e926dZsa00Y5bdiwITo6ev369evWrVu7du2aNWs+//zztWvXrl69+tNPP125cuWxY8c8Hg95dU0aqJYJDxVlsZCydXJEUTSZTGvXrt20aVN0dDS2m+UUGxu7ZcuWbdu27dixY9euXbt378YbQmNjY6Ojo7dt22az2bxeb3V1NcDHW0RRFueiEQ/pnzAB2+rVq7Ozs2HFkSNHtsopRiWhxujoaEAaHR390UcfHTt2TA2cIPnk2dOoDsscYgBDQjiO27t3b0xMzPbt27dt27br+rS7MR08eHDv3r1fyGnfvn27d+8+ePAgut5zc3PXr1+/devW2NjYzZs3x8TExMXFEYbAgxryli1b4uLitm3bFh0djaZrsVh27Nixb9++AwcOHJLTvn37Dh069OWXX+7fv/+AnPZfn/bu3bt169b8/HzW8OZGVLQAQIGWtqshoCGgIaAhoCGgIXADCJAvhQ7RlnqxCABOnDjRv3//Dz744P3333/nnXfefffdFStWvKWe3n333Xnz5o0dO7a4uFgQhLS0tFdffXX69OlRTFoip6WNacmSJYvlFBUVtWTJkuXLl/fp08dgMIiiWFNTM0tOK1eufO+9995n0j//+c/33nvvXTlBseXLly9btuydd94ZPHjw+vXrPR4PdaK3CMVAV5VFMkRRGHy4cOFC7969o6KiFixYADMXyWnx4sUwdunSpcvktHz58oULFy5atCgyMnLs2LElJSV+vz8nJ+eZZ56ZPHny3LlzZ8+ePX/+/AULFsyfPx+cixYtWiinBUyaN29ejx49zp49Cz0hDQUZrmtkVFTU0qVLlyxZsnTp0qioqEWLFg0ZMmTjxo0hmsmyocGw756hiTcAkGbRBB9awVnzeDwzZ84cNmzYlClTpk2bNjNomjFjxtSpUydNmjRw4MAhQ4ZAwunTp7t16zZ79mw0oTlz5sybN2+unObJaT6TFixYMGfOnOnTpz/77LN5eXkI3l577bWxY8dGRkZOk9OUKVNAT5kyZdKkSZMnT544ceKkSZMmymnSpEkREREvvPDC0aNHAQusbu66uxYAXD8IADiVW20EgG1yGq0hoCGgIaAhoCGgIXAdAnA+4AnBiQj+BdbrCss7KHX06NGxY8cWFhaazWZ82jLI1y3x7dHDhw9PnDixuLiY53mTydSn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5Q5XhctT5fHV+Py1Xl+Vz1PtdVfx3lreW8t5arzuKo+r0u2sdDkq7PZKl6tGkuCokX8Wp8vq9ti8Pic7HUi+M0i9C5wghQqNKwTobEpE42Jj6rIHn7R4R5ocFbAlL1fhuENOw1ZeuCyteZA7y1F3iFuMhoS+lYYmZdsUWzUJavxq+R5elG668j3Pzywfh2kNUDaAIt0lrwNX29EQ0BDQELglCLRtAECOcohEW5moVp2a/Jbyw6OVJgVIXdiSvy73QUq+fHV1ddfwsB3bt3B+2an1uwS/QxSdougSRZcgSL+G7iBp/NzPM68A8nkl55jzu70ehyj6fF4HPGOvxy5Iq28l39rtqvd5paN+yZOWXGe/3y4INq+3luOsouh0uSyW2gpLbaW1rtpSW2mplVwRi6WM5+2i6BBFB8/Xy1u7KHpqqssqyotLSwpKS4rKy0qKCvNLS4pKS4okBaQuQo8oOqWCcnehpbYyP89UUpxfkJ9TVJibl5udn2cqyDcJvIfnnYLg4jibKDoFweHzOsw5mcYsQ7YxzZhlyDFlZGXqzeb07Gy9dNRnFUU7x9kEQfqJoqO0KNuYcSUrU59tTMtIT33pxed/97vfHT9+3GKRPmcrCKLXK0144nx+zudPvZJy7ty548ePf/XVV19++eVXX3119OjRL774wmaz0WxynBS73Z6QkLB///6dctq3b9+ePXt27969Z88ecnapiMfjiZfTrl279u/fv2fPnsOHD++TE5oN60b7fL7Y2Njo6OgNGzZs2rQpOjp6jZyio6MVAxT4sPGncvroo49WymnVqlUrV65cvXp1fX093F+qQhCEf/3rXx9//PH777//wQcfvPfee//85z/ffffd7du319fXk1sMV57n+XXr1q1evfqjjz764IMPPv3007i4uIcffvj111+vrq5GqybHmuf51atXT506dfbs2dOmTZsyZcqkSZMmT548c+bMkpISsMEtFkXRarUuW7ZslJyGDRs2ZsyYESNGDBgwIDIy0mw2Q1va2my21atXjx49etSoUSNHjhw+fDgKfvDBBxiIoGuE53mXy7V69erx48dPnDhxTmOaO3fu+++/X11dDeebJFut1jVr1kyZMmXixImRkZFTpkyZNm3apEmT5s2bV1NTo5glVV1dvWrVqjlz5syfP3/x4sVRUVELFiyIior68MMPKXzCOmOe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/Zcjp479+8AH//zg3x8cOnDykzPfdV/SmEyIGzSYrzvz0KDb9emt3YPG764MfnWx7/Pvrhw89dm/D3/05rsHX3jjX0/++e29L76598/v7H3lr7tefOuJV97e+/o/HnvpjUdfeG3XS689++Zfz/X229w7ysH5P9d2AtwzAHx1w++n37kPxA+T7EvjHyZt6rFAulPnv12cvvLri36r6d7pfN2qnndan1stn6nze2nucrkMBgMKhfLz87vvvvs++OADZN8wcpr41IXftZw/rH6RWB75QgyAqzbA1VOAHM5hi/43fn7v//05aPis03jaAd9+dfAagH7fjemn8pw69Aect8cAmAj93ZR3Hbpbc9dMl6nk1K25br9Dux++Fs1wymb+Qi4i5gvyxkaHPQyAq0NjXu3Wo0Z+pV4A/ZFOOjo66ufnt3r1aoPBAG6tRjqyp+e2F9bPvV6QPj6px++ageXu9i7I5XCOD4/seWzHrIf84iNmJoRMT4+YRUlZyssMVZITankZHbLc9UX0zSrGWnluAzNNjgnixS8UJC6UoZeXk6ObuKg2EbZFjG3Nz2zkYeS4CE7aSlriEh4mRJAdIcBFi0kJAmICPzeJlZOYi45MjVkSEzovOmxeXOT85MSlsAGQFZpLiCLgIzIzgtJRS1Bpiwg5IUJBWn4+Si7HSmTpbG40mRqUS15OJC3LIQbCdwMTglnsBA43mZQbnp21gkgIp9PiuMxkASelREmsUtMba9hqVS6bHs1nxavkuAIBSiXH11dyGip4Un42JTM6KXx+fMjclJiFTFJifSWvrpK5ab3q8Uerduwof3RnTVFp7org3yehFmYTQ4j0SHZBep44JZcTy5WnC4uw4uIsuigxkwqvAkrNDpCXZivKshrbeJ0bZdWNLFUFtayKWdMkLq/m1zbJStQ8uYqhKmOXVPBqG6V1zdLGNkVds7R2laSxTdGyWtWyWtXcWbSqQ1leKyip5ChL8krK2fWrpE2tiromSW2jtLZJXl4jLqngl1UJq+plja3F7Wsq126q37i1ZfP2tvWbW9q7amsbi2VKLl9MFUqZMlV+QZGIzqPk8ak4cnZUYiSOnF1Urvznf/555sszn5397L0D773+1zc+OPTh1+fP9uk0317p/vzc+f3HT77xj3effuW1x5957vGnX3h037Nbdz+5Y99zO59+cfue5zc8/tTm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)
# 
# The figure shows CLIP training. Images of a sports car, a chef cooking, and hikers on a mountain pass through the **Image Encoder** to produce embedding vectors $I_1, I_2, I_3$. Matching text descriptions pass through the **Text Encoder** to produce text embedding vectors $T_1, T_2, T_3$.
# 
# CLIP computes the **dot product** between every image embedding and every text embedding, creating a similarity matrix. For two vectors $\mathbf{a}$ and $\mathbf{b}$, the dot product is defined as:
# 
# $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$
# 
# where $\theta$ is the angle between the vectors. When vectors are normalized to unit length, the dot product equals the **cosine similarity**, ranging from -1 for opposite directions to +1 for identical directions.
# 
# ### CLIP Training Objective
# 
# The diagonal elements of the similarity matrix represent correct image-text pairings. CLIP uses **contrastive learning** to push matching pairs together while pushing non-matching pairs apart. The loss has two symmetric components:
# 
# **Image-to-Text Loss**: For each image $I_i$, find its matching text among all texts in the batch:
# 
# $$\mathcal{L}_{I \rightarrow T} = -\log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_{j=1}^{N} \exp(I_i \cdot T_j / \tau)}$$
# 
# This is a softmax over all text options. The numerator measures similarity to the correct text $T_i$. The denominator sums similarity to all texts, acting as normalization. We want the correct pairing to dominate.
# 
# **Text-to-Image Loss**: For each text $T_i$, find its matching image among all images in the batch:
# 
# $$\mathcal{L}_{T \rightarrow I} = -\log \frac{\exp(T_i \cdot I_i / \tau)}{\sum_{j=1}^{N} \exp(T_i \cdot I_j / \tau)}$$
# 
# Same idea in reverse. Given a caption, which image does it describe?
# 
# The total loss averages both directions: $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{I \rightarrow T} + \mathcal{L}_{T \rightarrow I})$. The temperature $\tau$ controls how sharply the model distinguishes between similar and dissimilar pairs. This is known as **InfoNCE loss**. Training on 400 million image-text pairs from the internet, CLIP learns robust visual representations. It learns concepts like "sports car" or "chef cooking" from natural language rather than curated labels.
# 
# ### Why CLIP for Brain Decoding?
# 
# In Notebook 3, we were able to recover low-level visual structure, but the reconstructions were still blurry and often failed to resolve into a single, identifiable scene. That is not a bug in the model so much as an identifiability problem: many different natural images can match the same coarse edges, colors, and spatial layout. What we need next is a signal that disambiguates among those possibilities by providing category- and concept-level constraints.
# 
# CLIP is a strong target for that role because its embeddings emphasize high-level semantics rather than pixel-accurate detail, and they map images into a space where similarity reflects shared meaning. This makes CLIP a plausible match for what the human ventral stream represents, especially in higher-level visual cortex where object and scene identity dominates over fine-grained texture. If we can predict CLIP image embeddings from fMRI, we effectively recover the image’s “gist” as a compact semantic code. That semantic code can then act as a scaffold for generation. Instead of asking Stable Diffusion to invent an image from an ambiguous low-level cue alone, we condition it with a high-level guide that anchors identity and content.
# 
# 
# **Our goal in this notebook**: Train models that predict CLIP image embeddings from fMRI brain activity. Given a brain response $\mathbf{x} \in \mathbb{R}^{15724}$ from the nsdgeneral ROI recorded while viewing an image, we learn a mapping $f: \mathbf{x} \rightarrow \hat{\mathbf{y}}$ where $\hat{\mathbf{y}} \in \mathbb{R}^{1024}$ approximates the true CLIP embedding $\mathbf{y}$ of that image. We use **ViT-H/14** from OpenCLIP trained on LAION-2B as our target encoder.

# In[ ]:


from transformers import CLIPVisionModelWithProjection

print(f"Loading CLIP model: {hl.clip_id}...")
clip_model = CLIPVisionModelWithProjection.from_pretrained(hl.clip_id).to(device).eval()

# CLIP Normalization constants
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

@torch.no_grad()
def extract_clip_embeddings(images, batch_size=32, img_size=224):
    """
    Images: [N, 3, H, W] - can be uint8 [0-255] or float [0-1]
    Resizes to img_size on-the-fly to save memory (like NB3).
    Returns: [N, 1024] CLIP embeddings
    """
    resize = transforms.Resize((img_size, img_size), antialias=True)
    embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="CLIP Encoding"):
        batch = images[i:i+batch_size]

        # Convert uint8 to float [0,1] if needed
        if batch.dtype == torch.uint8:
            batch = batch.float() / 255.0

        # Resize to CLIP's expected size (224x224)
        batch = resize(batch)
        batch = batch.to(device)

        # Normalize for CLIP
        batch = (batch - mean) / std

        outputs = clip_model(batch)
        embeds = outputs.image_embeds
        embeddings.append(embeds.cpu())

    return torch.cat(embeddings, dim=0)

print("Extracting CLIP embeddings for Training Set...")
Ytr = extract_clip_embeddings(Itr)

print("Extracting CLIP embeddings for Validation Set...")
Yva = extract_clip_embeddings(Iva)

print("Extracting CLIP embeddings for Test Set...")
Yte = extract_clip_embeddings(Ite)

print(f"CLIP Targets Train Shape: {Ytr.shape}")
print(f"Sample Embedding Norm: {Ytr[0].norm().item():.2f}")

del clip_model
torch.cuda.empty_cache()


# ## 3. Training Our Models
# 
# Same approach as Notebook 3: normalize voxels and targets using training set statistics.

# In[ ]:


def zscore_train_apply(Xtr, Xva, Xte, eps=1e-6):
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, keepdim=True).clamp_min(eps)
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd, mu, sd

print("Normalizing voxels...")
mu_x = Xtr.mean(dim=0, keepdim=True)
sd_x = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)

Xtr     = (Xtr - mu_x) / sd_x
Xva     = (Xva - mu_x) / sd_x
Xte     = (Xte - mu_x) / sd_x
Xtr_exp = (Xtr_exp - mu_x) / sd_x    # same stats, different samples

print("Normalizing CLIP targets (for Ridge only)...")
Ytr, Yva, Yte, Ymu, Ysd = zscore_train_apply(Ytr, Yva, Yte)
print("Done.")


# # Part A: Ridge Regression
# 
# Same dual Ridge approach as Notebook 3, but predicting CLIP embeddings instead of VAE latents.

# In[ ]:


class DualRidge:
    def __init__(self, alpha: float = 1e5):
        self.alpha = alpha
        self.Xtr = None
        self.A = None

    def fit(self, Xtr, Ytr):
        Xtr = Xtr.float()
        Ytr = Ytr.float()
        N = Xtr.shape[0]
        K = Xtr @ Xtr.T
        K.diagonal().add_(self.alpha)
        self.A = torch.linalg.solve(K, Ytr)
        self.Xtr = Xtr
        return self

    def predict(self, X):
        X = X.float()
        return (X @ self.Xtr.T) @ self.A

print("--- Training Dual Ridge ---")
print("Tuning Alpha on Validation Set...")

best_mse = float("inf")
best_alpha = hl.ridge_alphas[0]

for a in hl.ridge_alphas:
    ridge = DualRidge(alpha=a).fit(Xtr, Ytr)
    Yva_pred = ridge.predict(Xva)
    mse = torch.mean((Yva_pred - Yva)**2).item()
    print(f"Alpha: {a} | Val MSE: {mse:.4f}")
    if mse < best_mse:
        best_mse = mse
        best_alpha = a

print(f"\nBest Alpha: {best_alpha}")

# Final Ridge Fit
print("Training Final Ridge Model...")
ridge = DualRidge(alpha=best_alpha).fit(Xtr, Ytr)
Yte_pred_ridge = ridge.predict(Xte)
print("Ridge training complete.")


# ## Ridge Evaluation: Retrieval Accuracy
# 
# For predicted CLIP embeddings, MSE can be a weak signal because we do not actually care about matching every coordinate perfectly. What we care about is whether the prediction lands in the right *region* of CLIP space so that it is closest to the correct image embedding.
# 
# In our implementation, we first L2-normalize the predicted and target embeddings, so similarity is measured by cosine similarity, which is essentially the dot product after normalization:
# 
# $$
# \mathrm{sim}(\mathbf{p}, \mathbf{t}) = \mathbf{p}^\top \mathbf{t}
# $$
# 
# We then compute an all-to-all similarity matrix where each row corresponds to one predicted embedding and each column corresponds to one candidate ground-truth image embedding.
# 
# **Top-1 accuracy** asks a strict retrieval question: for each brain prediction (row), is the most similar image the correct one? This is evaluated on the first 300 test samples, so random chance is 0.33%.
# 
# **2-Way forced choice** pairs each sample against a single random distractor and checks if the correct image wins. Averaged over 1000 random trials for stability. Random chance is 50%.
# 
# **Pairwise accuracy** generalizes the 2-way test to *all* distractors: for each trial $i$, we count how often the correct similarity $\mathrm{sim}_{i,i}$ beats every incorrect $\mathrm{sim}_{i,j}$, then average. This is the probability that the model ranks the true image above a randomly chosen distractor.
# 
# Together, these metrics tell us whether the predictions preserve the semantic neighborhood structure of CLIP space.

# In[ ]:


def evaluate_retrieval(pred_features, target_features, name="Model", n_eval=300):
    # Evaluate on first n_eval samples (smaller pool = cleaner retrieval metric)
    pred_features = pred_features[:n_eval]
    target_features = target_features[:n_eval]

    pred_norm = torch.nn.functional.normalize(pred_features, dim=1)
    target_norm = torch.nn.functional.normalize(target_features, dim=1)
    sim_matrix = pred_norm @ target_norm.T
    n = sim_matrix.shape[0]

    labels = torch.arange(n).to(sim_matrix.device)
    top1_acc = (sim_matrix.argmax(dim=1) == labels).float().mean().item()

    # Pairwise accuracy
    correct_sims = sim_matrix.diag().view(-1, 1)
    pairwise_wins = (correct_sims > sim_matrix).float().sum()
    pairwise_acc = pairwise_wins / (n * (n - 1))

    # 2-way forced choice: for each sample, pick a random distractor and check
    # if the correct target scores higher. Average over many trials for stability.
    n_trials = 1000
    torch.manual_seed(42)
    wins = 0
    for _ in range(n_trials):
        # Random distractor index for each sample (not equal to the correct one)
        distractors = torch.randint(0, n - 1, (n,))
        distractors[distractors >= labels] += 1
        distractor_sims = sim_matrix[torch.arange(n), distractors]
        wins += (sim_matrix.diag() > distractor_sims).float().sum().item()
    two_way_acc = wins / (n * n_trials)

    print(f"{name} Results (N={n}):")
    print(f"  Top-1 Accuracy:      {top1_acc*100:.2f}%")
    print(f"  2-Way Forced Choice: {two_way_acc*100:.2f}%")
    print(f"  Pairwise Accuracy:   {pairwise_acc*100:.2f}%")
    return top1_acc, pairwise_acc, two_way_acc

# Unnormalize Ridge predictions
Yte_pred_ridge_un = Yte_pred_ridge * Ysd + Ymu
Yte_un = Yte * Ysd + Ymu

print("=== Ridge Results ===")
ridge_top1, ridge_pairwise, ridge_2way = evaluate_retrieval(Yte_pred_ridge_un, Yte_un, name="Ridge")

def show_retrieval(pred_un, model_name, test_images, target_un):
    """Show retrieval results: GT image vs retrieved image for sample indices."""
    print(f"{model_name}: Brain's Best Guess (Retrieval)")

    indices = [0, 10,20,30,40, 50, 100, 200]
    if len(pred_un) < 200: indices = [0, 1, 2]

    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4*len(indices)))

    for i, idx in enumerate(indices):
        gt_img = test_images[idx].permute(1, 2, 0).numpy()

        pred_vec = pred_un[idx].unsqueeze(0)
        sims = torch.nn.functional.cosine_similarity(pred_vec, target_un)
        best_idx = sims.argmax().item()
        ret_img = test_images[best_idx].permute(1, 2, 0).numpy()

        ax_gt = axes[i, 0] if len(indices) > 1 else axes[0]
        ax_ret = axes[i, 1] if len(indices) > 1 else axes[1]

        ax_gt.imshow(gt_img)
        ax_gt.set_title(f"GT Image {idx}")
        ax_gt.axis("off")

        ax_ret.imshow(ret_img)
        ax_ret.set_title(f"{model_name} Retrieved (Sim: {sims[best_idx]:.2f})")
        ax_ret.axis("off")

    plt.suptitle(f"{model_name}: Retrieval Results", fontsize=14)
    plt.tight_layout()
    plt.show()


# ## Ridge Visualization
# 
# We retrieve the single most similar test image by selecting the one with the highest cosine similarity to the predicted Ridge embedding.
# 

# In[ ]:


show_retrieval(Yte_pred_ridge_un, "Ridge", Ite, Yte_un)


# # Part B: MLP
# 
# In Part A we used Ridge regression as a linear baseline. Now we train a nonlinear MLP with a different training objective motivated by our downstream pipeline. In Notebook 5, IP-Adapter normalizes the predicted CLIP embedding before using it, so it only sees the direction of the vector, not its magnitude. A standard MSE loss wastes capacity learning magnitudes that get discarded at generation time.
# 
# We fix this with two changes. First, our `CLIPMLP` applies L2 normalization as its final operation, constraining every output to the unit hypersphere. Second, we replace MSE with two complementary losses from the brain decoding literature. **Cosine loss** directly maximizes the directional alignment between prediction and target, ignoring magnitude entirely. **SoftCLIP loss** adds a structural constraint: within each batch, ground-truth CLIP embeddings define a pairwise similarity matrix capturing which images are semantically close, and our model must reproduce that relational structure using its own predictions via KL divergence. This prevents collapse toward the batch mean and preserves the fine-grained semantic distinctions that retrieval and generation depend on. We also train on expanded voxel repeats rather than trial-averaged data for our MLP again.

# In[ ]:


import torch.nn.functional as F

class CLIPMLP(nn.Module):
    """Maps voxels onto the CLIP embedding hypersphere."""

    def __init__(self, in_dim, out_dim, hidden_dims=(2048, 2048), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def cosine_loss(pred, target):
    """1 - mean cosine similarity. pred is already unit-length."""
    target_n = F.normalize(target, dim=-1)
    return 1.0 - (pred * target_n).sum(dim=-1).mean()


def soft_clip_loss(pred, target, temperature=0.07):
    """SoftCLIP distillation (MindEye-style).

    Teacher signal: pairwise similarity matrix of ground-truth CLIP
    embeddings within the batch. Student must reproduce that structure
    using its own predictions. Forces the model to preserve the
    neighborhood geometry of CLIP space, not just minimize per-sample
    error.
    """
    target_n = F.normalize(target, dim=-1)

    teacher_logits = target_n @ target_n.T / temperature
    teacher_probs  = F.softmax(teacher_logits, dim=-1)

    student_logits    = pred @ target_n.T / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


# In[ ]:


print("Training MLP")
set_seed(hl.seed)

# Targets: raw CLIP embeddings, expanded to match voxel repeats
Ytr_clip = (Ytr * Ysd + Ymu).float()                  # unnormalize
Yva_clip = (Yva * Ysd + Ymu).float()

Ytr_clip_exp = Ytr_clip.repeat_interleave(R, dim=0)   # match expanded voxels
print(f"Expanded targets: {Ytr_clip_exp.shape[0]} (should match {Xtr_exp.shape[0]})")
assert Ytr_clip_exp.shape[0] == Xtr_exp.shape[0]

# Model
mlp = CLIPMLP(
    in_dim=Xtr_exp.shape[1],
    out_dim=Ytr_clip.shape[1],
    hidden_dims=hl.mlp_hidden_dims,
    dropout=hl.mlp_dropout,
).to(device)

n_params = sum(p.numel() for p in mlp.parameters())
print(f"Architecture: (2048, 2048) -> L2 normalize")
print(f"Parameters: {n_params:,}")

# Hyperparameters
MLP_LR       = hl.mlp_lr
MLP_WD       = hl.mlp_weight_decay
MLP_EPOCHS   = hl.mlp_epochs
MLP_BATCH    = hl.mlp_batch_size
NOISE_STD    = hl.mlp_noise_std
COS_WEIGHT   = hl.mlp_cos_weight
SC_WEIGHT    = hl.mlp_softclip_weight
SC_TEMP      = hl.mlp_softclip_temp
PATIENCE     = hl.mlp_patience

optimizer = optim.AdamW(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MLP_EPOCHS, eta_min=MLP_LR * 0.01
)

# Dataloaders (expanded data for MLP)
Xtr_dev = Xtr_exp.float().to(device)
Ytr_dev = Ytr_clip_exp.to(device)
Xva_dev = Xva.float().to(device)
Yva_dev = Yva_clip.to(device)

train_ds = torch.utils.data.TensorDataset(Xtr_dev, Ytr_dev)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=MLP_BATCH, shuffle=True, drop_last=True
)

#Training
best_val_cos  = -1.0
best_state    = None
patience_ctr  = 0
train_losses  = []
val_cosines   = []

print(f"Training for up to {MLP_EPOCHS} epochs  (patience={PATIENCE})...")
print(f"Loss: {COS_WEIGHT}*cosine + {SC_WEIGHT}*SoftCLIP  (temp={SC_TEMP})")

for epoch in range(MLP_EPOCHS):
    mlp.train()
    epoch_loss, n_seen = 0.0, 0

    for bx, by in train_dl:
        bx = bx + torch.randn_like(bx) * NOISE_STD

        optimizer.zero_grad(set_to_none=True)
        pred = mlp(bx)

        loss = (COS_WEIGHT * cosine_loss(pred, by)
              + SC_WEIGHT  * soft_clip_loss(pred, by, temperature=SC_TEMP))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * bx.size(0)
        n_seen    += bx.size(0)

    scheduler.step()
    train_losses.append(epoch_loss / n_seen)

    # Validation: cosine similarity on averaged voxels
    mlp.eval()
    with torch.no_grad():
        pred_va  = mlp(Xva_dev)
        va_tgt_n = F.normalize(Yva_dev, dim=-1)
        val_cos  = (pred_va * va_tgt_n).sum(dim=-1).mean().item()
    val_cosines.append(val_cos)

    if val_cos > best_val_cos:
        best_val_cos = val_cos
        best_state   = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:3d}/{MLP_EPOCHS}  "
              f"Loss {train_losses[-1]:.4f}  Val cos {val_cos:.4f}")

    if patience_ctr >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}")
        break

mlp.load_state_dict(best_state)
mlp = mlp.to(device).eval()

with torch.no_grad():
    Yte_pred_mlp = mlp(Xte.float().to(device)).cpu()

print(f"\nBest validation cosine similarity: {best_val_cos:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses);  ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training Loss (Cosine + SoftCLIP)")
ax2.plot(val_cosines);   ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cosine Sim")
ax2.set_title("Validation Cosine Similarity")
plt.tight_layout(); plt.show()

# Free expanded data from GPU
del Xtr_dev, Ytr_dev, Xtr_exp, Ytr_clip_exp
torch.cuda.empty_cache()


# ## MLP Evaluation
# 
# We evaluate the MLP using the same retrieval metrics as Ridge.

# In[ ]:


# MLP outputs unit vectors. Scale using TRAIN-set CLIP norms so downstream
# code and saving get embeddings in the same ballpark as Ridge and GT
# without using any test-set statistics.
train_mean_norm = Ytr_clip.norm(dim=-1).mean()
Yte_pred_mlp_un = Yte_pred_mlp * train_mean_norm

print("=== MLP Results ===")
mlp_top1, mlp_pairwise, mlp_2way = evaluate_retrieval(
    Yte_pred_mlp_un, Yte_un, name="MLP"
)

print("\n=== Comparison ===")
print(f"Ridge  | Top-1: {ridge_top1*100:.1f}%  "
      f"| 2-Way: {ridge_2way*100:.1f}%  "
      f"| Pairwise: {ridge_pairwise*100:.1f}%")
print(f"MLP    | Top-1: {mlp_top1*100:.1f}%  "
      f"| 2-Way: {mlp_2way*100:.1f}%  "
      f"| Pairwise: {mlp_pairwise*100:.1f}%")


# ## Full Retrival Results
# 
# We visualize the MLP's retrieval results with Ridge retrieval results and ground truth images.
# 

# In[ ]:


def show_retrieval_comparison(ridge_un, mlp_un, test_images, target_un, n_show=25, n_eval=300):
    """Show GT vs Ridge retrieval vs MLP retrieval in three straight columns."""
    pool_ridge  = ridge_un[:n_eval]
    pool_mlp    = mlp_un[:n_eval]
    pool_target = target_un[:n_eval]
    pool_images = test_images[:n_eval]

    indices = list(range(0, n_eval, n_eval // n_show))[:n_show]

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    fig.suptitle("Retrieval Comparison: GT | Ridge | MLP", fontsize=14, y=1.005)

    for i, idx in enumerate(indices):
        gt_img = pool_images[idx].permute(1, 2, 0).numpy()

        ridge_sims = F.cosine_similarity(pool_ridge[idx].unsqueeze(0), pool_target)
        ridge_best = ridge_sims.argmax().item()
        ridge_img  = pool_images[ridge_best].permute(1, 2, 0).numpy()

        mlp_sims = F.cosine_similarity(pool_mlp[idx].unsqueeze(0), pool_target)
        mlp_best = mlp_sims.argmax().item()
        mlp_img  = pool_images[mlp_best].permute(1, 2, 0).numpy()

        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f"GT {idx}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ridge_img)
        axes[i, 1].set_title(f"Ridge (sim={ridge_sims[ridge_best]:.2f})", fontsize=8)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mlp_img)
        axes[i, 2].set_title(f"MLP (sim={mlp_sims[mlp_best]:.2f})", fontsize=8)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

show_retrieval_comparison(Yte_pred_ridge_un, Yte_pred_mlp_un, Ite, Yte_un)


# Two patterns stand out from the retrieval results. First, when retrieval fails it tends to fail gracefully. The retrieved image is rarely a complete semantic mismatch. A beach scene might retrieve a different beach, or a dog might retrieve a different animal, suggesting that the predicted embeddings land in roughly the right neighborhood of CLIP space even when they miss the exact target. This is encouraging for downstream reconstruction, where getting the general category right matters more than identifying the precise exemplar. Second, the MLP consistently retrieves more semantically faithful matches than Ridge, confirming what the quantitative metrics already suggested: a nonlinear model with contrastive training maps the voxel space into CLIP space more accurately than a linear baseline.

# ## 4. Generative Reconstruction
# 
# We generate images using **Stable Diffusion 1.5** with **IP-Adapter** as a downstream sanity check on our decoded embeddings. After predicting a 1024-D OpenCLIP embedding from fMRI, we feed that embedding into IP-Adapter to condition the diffusion model. If the embedding is accurate, the generated images should preserve the correct high-level content (objects and scene identity), even if low-level details vary. This step is not meant to perfectly reproduce the original stimulus, but to qualitatively test whether our predicted CLIP space captures the right semantic “gist.”
# 
# 
# 

# In[ ]:


from diffusers import StableDiffusionPipeline

print("Loading SD 1.5 + IP-Adapter (compatible with 1024-dim OpenCLIP)...")

# 1. Load Base SD 1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
).to(device)

# 2. Load IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

# Optimization
pipe.enable_model_cpu_offload()


# In[ ]:


@torch.no_grad()
def generate_from_clip(clip_embedding, seed=42):
    """
    Generates image from 1024-dim embedding using IP-Adapter.
    """
    # Reshape: [1, 1024] -> [1, 1, 1024]
    valid_embeds = clip_embedding.view(1, 1, -1).to(device, dtype=torch.float16)

    # Create negative embeddings (zeros) to match positive shape
    neg_embeds = torch.zeros_like(valid_embeds)

    # Concatenate [negative, positive] for Classifier-Free Guidance
    combined_embeds = torch.cat([neg_embeds, valid_embeds], dim=0)

    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        prompt="best quality, high quality",
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ip_adapter_image_embeds=[combined_embeds],
        num_inference_steps=30,
        generator=generator,
    ).images

    return images[0]


# In[ ]:


# Generate from a few test samples
print("Generating images from brain-predicted CLIP embeddings...")
sample_indices = [0,20, 50, 70, 100]

fig, axes = plt.subplots(len(sample_indices), 3, figsize=(10, 4*len(sample_indices)))

for i, idx in enumerate(sample_indices):
    gt_img = Ite[idx].permute(1, 2, 0).numpy()
    axes[i, 0].imshow(gt_img)
    axes[i, 0].set_title(f"Ground Truth {idx}")
    axes[i, 0].axis("off")

    ridge_gen = generate_from_clip(Yte_pred_ridge_un[idx:idx+1])
    axes[i, 1].imshow(ridge_gen)
    axes[i, 1].set_title("Ridge Generated")
    axes[i, 1].axis("off")

    mlp_gen = generate_from_clip(Yte_pred_mlp_un[idx:idx+1])
    axes[i, 2].imshow(mlp_gen)
    axes[i, 2].set_title("MLP Generated")
    axes[i, 2].axis("off")

plt.suptitle("Brain → CLIP → Generated Image", fontsize=14)
plt.tight_layout()
plt.show()


# The generated images capture the broad semantic content of the original stimuli reasonably well, particularly for scenes with strong categorical signals. Keep in mind that we are conditioning a frozen generative model with a single 1024-dimensional vector predicted from noisy brain data using a simple two-layer MLP. The fact that this produces recognizable and often category-correct outputs speaks to both the richness of the fMRI signal and the power of CLIP as a representational target.

# # Saving Results
# 
# We save CLIP embeddings for potential combination with low-level results (from Notebook 3) in future work.

# In[ ]:


# Unnormalize Train/Val/Test targets for saving
Ytr_un = Ytr * Ysd + Ymu
Yva_un = Yva * Ysd + Ymu

# Create separate dictionaries for each split
train_dict = {
    "gt_clip_train": Ytr_un.cpu()
}

val_dict = {
    "gt_clip_val": Yva_un.cpu()
}

test_dict = {
    "pred_clip_ridge": Yte_pred_ridge_un.cpu(),
    "pred_clip_mlp": Yte_pred_mlp_un.cpu(),
    "gt_clip_test": Yte_un.cpu(),
    "test_images": Ite.cpu(),
    "ridge_metrics": {"top1": ridge_top1, "pairwise": ridge_pairwise, "two_way": ridge_2way},
    "mlp_metrics": {"top1": mlp_top1, "pairwise": mlp_pairwise, "two_way": mlp_2way},
}

# Ensure directory exists before saving
try:
    os.makedirs(hl.root, exist_ok=True)
    torch.save(train_dict, os.path.join(hl.root, "high_level_train_clips.pt"))
    torch.save(val_dict, os.path.join(hl.root, "high_level_val_clips.pt"))
    torch.save(test_dict, os.path.join(hl.root, "high_level_test_results.pt"))
    print(f"Saved Train/Val/Test results to: {hl.root}")
except Exception as e:
    print(f"Could not save to {hl.root}: {e}")


# ## What's Next
# 
# In this notebook, we built a high-level semantic pipeline that predicts CLIP embeddings from brain activity. These 1024-dimensional vectors capture what the brain "thinks" it's seeing in terms of objects, scenes, and meaning. Our retrieval experiments demonstrated that the brain encodes rich semantic information that can be decoded with relatively simple models.
# 
# However, CLIP embeddings alone cannot produce pixel-accurate reconstructions. They tell us *what* is in an image but not *where* things are or *what colors* they have. This is the inverse limitation of Notebook 3, which captured spatial structure but lacked semantic awareness.
# 
# [In Notebook 5: Hybrid Reconstructions](https://colab.research.google.com/drive/1nl-20PpwWA7KBP1w6A8QESuVO_uvAgen), we combine both approaches into a hybrid reconstruction pipeline. The key insight is that these two representations are complementary: low-level VAE latents provide spatial structure, while high-level CLIP embeddings provide semantic guidance. By feeding our blurry Notebook 3 reconstructions into an image-to-image diffusion model and conditioning on our predicted CLIP embeddings via IP-Adapter, we can generate reconstructions that are both structurally accurate and semantically correct.

# ## Source: Notebook5_HybridConstruction

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


print("Loading SDXL Components (safe IP-Adapter setup)...")
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

# In[76]:


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
    seed=12,
    pass1_steps=24,
    pass2_steps=28,
    pass1_strength=0.62,
    pass2_strength=0.35,
):
    pil_low = preprocess_image(low_level_img_tensor)
    pil_low = pil_low.resize((1024, 1024), Image.LANCZOS)

    combined_embeds = prepare_ip_adapter_embeds(embedding_tensor)

    # Pass 1: semantic correction while preserving low-level layout
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.set_ip_adapter_scale(0.9)
    pass1_result = pipe(
        prompt="realistic",
        negative_prompt="blurry, unrealistic, extra limbs",
        image=pil_low,
        ip_adapter_image_embeds=[combined_embeds],
        strength=pass1_strength,
        num_inference_steps=pass1_steps,
        guidance_scale=5.26,
        generator=generator,
    ).images[0]

    # Pass 2: refinement directly on pass1 output
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.set_ip_adapter_scale(0.3)
    final_result = pipe(
        prompt="high quality, highly detailed, sharp focus, realistic, hyperrealistic",
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
    pipe.set_ip_adapter_scale(1.0)
    result = pipe(
        prompt="",
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

# In[77]:


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

# In[78]:


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

# In[79]:


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

# In[80]:


N_QUANT_EVAL = 30
#N_QUANT_EVAL = len(gt_images_tensor) #Uncomment this line to run the full evaluation, same with the paper.
SEM_ONLY_STEPS = 20
HYB_PASS1_STEPS = 24
HYB_PASS2_STEPS = 28
BASE_SEED = 4

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

# In[81]:


# Visualize a grid of reconstructions
N_SHOW = N_QUANT_EVAL
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


# The reconstructions illustrate the complementary nature of the two signals. Low-level outputs preserve coarse layout and color but lack semantic content. Semantic-only outputs often identify the correct category but invent the spatial context entirely. Hybrid outputs combine both, placing recognizable objects in approximately correct positions with approximately correct colors, though fine details diverge from the ground truth. The combination does not benefit every sample equally, and this variability is more visible in our pipeline than in state-of-the-art systems because we lack the dedicated diffusion prior and custom generation model that systems like MindEye use to bridge the gap between noisy brain-predicted embeddings and the conditioning interface of the generative model.
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
