#!/usr/bin/env python
# coding: utf-8

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
