import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from torch import nn
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from natsort import natsorted
import sys
import torch.nn as nn
import sys
sys.path.append("/home/aleximu/gunes/dinov2")
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math


# Define the model class
class DinoVisionTransformerSegmentation(nn.Module):
    def __init__(self, model_size="base", num_labels=2):
        super(DinoVisionTransformerSegmentation, self).__init__()
        self.model_size = model_size

        n_register_tokens = 4

        if model_size == "small":
            model = vit_small(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 384
            self.number_of_heads = 6

        elif model_size == "base":
            model = vit_base(patch_size=14,
                             img_size=526,
                             init_values=1.0,
                             num_register_tokens=n_register_tokens,
                             block_chunks=0)
            self.embedding_size = 768
            self.number_of_heads = 12

        elif model_size == "large":
            model = vit_large(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 1024
            self.number_of_heads = 16

        elif model_size == "giant":
            model = vit_giant2(patch_size=14,
                               img_size=526,
                               init_values=1.0,
                               num_register_tokens=n_register_tokens,
                               block_chunks=0)
            self.embedding_size = 1536
            self.number_of_heads = 24

        self.transformer = model

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.embedding_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_labels, kernel_size=1)
        )

    def forward(self, pixel_values):
        transformer_output = self.transformer.forward_features(pixel_values)

        patch_embeddings = transformer_output["x_norm_patchtokens"]
        batch_size = patch_embeddings.size(0)
        sequence_length = patch_embeddings.size(1)
        embedding_size = patch_embeddings.size(2)

        patch_size = int(math.sqrt(sequence_length))
        patch_embeddings = patch_embeddings.permute(0, 2, 1).contiguous().view(batch_size, embedding_size, patch_size, patch_size)
        head_output = self.segmentation_head(patch_embeddings)

        segmentation_output = F.interpolate(head_output, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        return segmentation_output

# Paths
input_image_path = "/home/aleximu/gunes/dinov2/project/dataset/fishency/train/fishes/img_0001_00000.png"
output_image_path = "/home/aleximu/gunes/dinov2/outputs_video/segmented_image.png"

# Load the model and weights
model = DinoVisionTransformerSegmentation("base")
pretrained_path = "/home/aleximu/gunes/dinov2/project/ssl_best_model1(80).pth"
state_dict = torch.load(pretrained_path)
model.load_state_dict(state_dict, strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transformation
target_size = (448, 448)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Function to segment an image
def segment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    transformed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        logits = output
        probs = torch.softmax(logits, dim=1)
        predicted_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    return predicted_mask, original_image, transformed_image

# Segment the input image
segmented_mask, original_image, transformed_image = segment_image(input_image_path)

# Debugging: Check unique values in the segmented mask
print("Unique values in the segmented mask:", np.unique(segmented_mask))

# Debugging: Visualize the transformed input image
transformed_image_np = transformed_image.squeeze().permute(1, 2, 0).cpu().numpy() * STD + MEAN
transformed_image_np = (transformed_image_np * 255).astype(np.uint8)

# Convert the segmented mask to a displayable format
segmented_mask_image = Image.fromarray((segmented_mask * 255).astype(np.uint8))

# Visualize the original image, transformed image, and segmented mask
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)

plt.subplot(1, 3, 2)
plt.title("Transformed Image")
plt.imshow(transformed_image_np)

plt.subplot(1, 3, 3)
plt.title("Segmented Mask")
plt.imshow(segmented_mask_image, cmap='jet', alpha=0.6)
plt.show()

# Save the output
segmented_mask_image.save(output_image_path)
print("Image segmentation completed and saved to", output_image_path)