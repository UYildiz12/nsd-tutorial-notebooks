# NSD Tutorial Notebooks

A hands-on tutorial series for reconstructing visual images from human brain activity using the Natural Scenes Dataset (NSD). These notebooks walk through the full pipeline, from raw fMRI signals to photorealistic image reconstructions, following the same principles that underpin state-of-the-art methods like MindEye and Brain-Diffuser. The entire series runs on Google Colab's free tier with a T4 GPU. All code, data streaming, and model loading are optimized to fit within 16GB of VRAM and standard Colab RAM limits, so no paid compute or local hardware is needed.

## What This Is

When a person looks at a photograph, their visual cortex produces rich patterns of neural activity. Different brain regions encode different aspects of the scene: early visual areas respond to edges, colors, and spatial layout, while higher areas represent objects, faces, and semantic categories. This tutorial series shows how to read those patterns out of fMRI data and turn them back into images.

The reconstruction pipeline has three stages. First, we learn to predict low-level visual features (VAE latents) from brain activity, producing blurry reconstructions that capture spatial structure and color. Second, we predict high-level semantic features (CLIP embeddings) that describe what objects and scenes are present. Third, we fuse both signals through a diffusion model that adds realistic detail consistent with the predicted content. The result is a reconstructed image built entirely from brain data.

## Notebooks

### Notebook 0 -- Basics

Covers the Colab environment, GPU setup, and package management. Start here if you are new to running notebooks in the cloud.

### Notebook 1 -- fMRI

Introduces the neuroscience foundations: what fMRI measures, how the BOLD signal works, and how voxels tile the brain. Walks through the NSD dataset, including data loading via WebDataset streaming, the role of the `nsdgeneral` ROI mask, and why averaging across stimulus repetitions reduces noise.

### Notebook 2 -- VAEs

Explains Variational Autoencoders and why they matter for brain decoding. Demonstrates encoding and decoding real images through the Stable Diffusion VAE, illustrating how a high-dimensional image can be compressed into a compact latent representation that preserves perceptual structure.

### Notebook 3 -- Low-Level Pipeline

Builds the first decoding model. Maps voxel activity patterns to VAE latents using Ridge regression and MLP models, then decodes those latents back into images. Covers data preparation, target normalization, training with spatial gradient loss, and evaluation with SSIM. The resulting reconstructions are blurry but spatially coherent.

### Notebook 4 -- High-Level Pipeline

Builds the semantic decoding model. Encodes images into OpenCLIP embedding space, then trains models to predict these embeddings from brain activity. The core evaluation metric is retrieval accuracy: given a predicted embedding, can we pick the correct image out of a lineup? This notebook demonstrates that the brain encodes enough semantic information to reliably identify viewed content.

### Notebook 5 -- Hybrid Reconstruction

Brings everything together. Uses Stable Diffusion XL in image-to-image mode, feeding the blurry low-level reconstruction as structural input and the predicted CLIP embedding as semantic guidance through an IP-Adapter. The diffusion model synthesizes a final image that is both spatially faithful and semantically appropriate, producing results far better than either signal achieves alone.

## Repository Structure

```
Notebooks/
  CurrentTutorialPipeline/
    Notebook0_Basics.ipynb
    Notebook1_fMRI.ipynb
    Notebook2_VAEs.ipynb
    Notebook3_LowLevelPipeline.ipynb
    Notebook4_HighLevelPipeline.ipynb
    Notebook5_HybridConstruction.ipynb
  Py_versions/          # Synced .py exports, including the merged tutorial export
```

## Getting Started

1. Open a notebook from `CurrentTutorialPipeline/` in Google Colab and select a T4 GPU runtime.
2. Run cells sequentially. Each notebook installs its own dependencies and streams data directly from Hugging Face Hub.
3. Notebooks 3 through 5 build on results from earlier notebooks, so work through them in order.

## References

- Scotti et al. (2023). *MindEye: fMRI-to-Image Reconstruction with Retrieval Augmented Diffusion.* arXiv:2305.10865
- Ozcelik and VanRullen (2023). *Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion.* arXiv:2303.05334
- Scotti et al. (2024). *MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data.* ICML 2024
- Allen et al. (2022). *A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence.* Nature Neuroscience
