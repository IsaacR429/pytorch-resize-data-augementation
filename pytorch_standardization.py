import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
from torchvision.utils import save_image

# Define the augmentation pipeline
fabrics_data_augmentation_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=45),
    v2.ColorJitter(brightness=0.6, contrast=0.6),
    v2.RandomZoomOut(fill=255, side_range=(1.0, 1.2), p=0.5),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#  Load dataset
dataset_images = datasets.ImageFolder("Fabrics_image", transform=fabrics_data_augmentation_transforms)
dataloader = DataLoader(dataset_images, batch_size=1, shuffle=True)

output_root = "augmented_images_pytorch"
num_augments = 4  # <-- Number of variations per image

#  Create augmented versions
for aug_idx in range(num_augments):
    for idx, (img_tensor, label) in enumerate(dataloader):
        class_name = dataset_images.classes[label.item()].lower().replace(" ", "_")
        save_dir = os.path.join(output_root, class_name)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{class_name}_aug_{idx}_v{aug_idx}.png"
        save_path = os.path.join(save_dir, filename)

        # Unnormalize to [0, 1] range for saving
        img_tensor = img_tensor * 0.5 + 0.5
        save_image(img_tensor, save_path)

        print(f" Saved: {save_path}")
