# Fabric Image Preprocessing Pipeline (PyTorch)

This project implements an image standardization and augmentation pipeline using **PyTorch** and **Torchvision** for a cultural fabric recognition dataset. The dataset includes traditional fabrics from **Burkina Faso**, **Ghana**, and **India**, organized into subfolders for each country.

---

##  Project Structure

Fabrics_image/
├── Burkina_Faso_Fabrics/
├── Ghana_Fabrics/
├── India_Fabrics/

augmented_images_pytorch/
├── burkina_faso_fabrics/
├── ghana_fabrics/
├── india_fabrics/


##  Purpose

To prepare images for deep learning classification tasks (e.g., predicting the **country of origin**) using CNNs by applying:

- Standardization: consistent shape, type, and scale
- Data Augmentation: increase dataset diversity and robustness

---

##  Pipeline Overview

### Step 1: Define PyTorch Transforms (using `torchvision.transforms.v2`)


## Standardization & Augmentation

### Objective:

Ensure all images are:

- Converted to PyTorch tensors
- Normalized to a common scale
- Resized to 224×224

Augmented for robustness

- Transformations Applied:
- Using torchvision.transforms.v2:
- ToImage – Convert PIL images to tensors
- RandomHorizontalFlip (p=0.5) – Simulates fabric from different viewpoints
- RandomRotation (±45°) – Helps generalize to rotated patterns
- ColorJitter (brightness & contrast) – Simulates lighting variation
- RandomZoomOut – Adds padding-based zoom effect
- Resize to (224, 224) – Standard input shape
- Normalize – Scale pixel values to [-1, 1]


## Tools used

- torch for tensors and data loading
- torchvision.datasets.ImageFolder to load structured dataset
- torchvision.transforms.v2 for augmentation
- save_image to write images to disk