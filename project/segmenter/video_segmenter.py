import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import torch.nn as nn
import math
import torch.nn.functional as F
from pathlib import Path

# Add the project directory to the system path
sys.path.append("/home/aleximu/gunes/dinov2")

# Import the vision transformer models
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

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
input_video_path = "/home/aleximu/gunes/dinov2/project/videos/output_video_part1.mp4"
output_video_folder = "/home/aleximu/gunes/dinov2/outputs_video/"
output_video_filename = "segmented_output_video.mp4"
output_video_path = os.path.join(output_video_folder, output_video_filename)

# Load model
model = DinoVisionTransformerSegmentation("base")
pretrained_path = "/home/aleximu/gunes/dinov2/project/ssl_best_model1(80).pth"
state_dict = torch.load(pretrained_path)
model.load_state_dict(state_dict, strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transformation settings
target_size = (448, 448)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Function to segment a frame
def segment_frame(frame):
    image = Image.fromarray(frame)
    transformed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        logits = output
        probs = torch.softmax(logits, dim=1)
        predicted_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    # Convert the predicted mask to 0 (background) and 255 (fish)
    predicted_mask_scaled = (predicted_mask * 255).astype(np.uint8)

    return predicted_mask_scaled

# Read the video
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Process each frame
        segmented_frame = segment_frame(frame)

        # Ensure the segmented frame is in 0-255 range
        segmented_frame_display = segmented_frame.astype(np.uint8)

        # Resize the mask to match the original frame size
        segmented_frame_resized = cv2.resize(segmented_frame_display, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert the mask to a 3-channel image for combining
        segmented_frame_color = cv2.cvtColor(segmented_frame_resized, cv2.COLOR_GRAY2BGR)

        # Combine the original frame and the segmented mask for visualization
        combined_frame = cv2.addWeighted(frame, 0.7, segmented_frame_color, 0.3, 0)

        # Write the frame to the output video
        out.write(combined_frame)
    else:
        break

cap.release()
out.release()

print("Video segmentation completed and saved to ---->", output_video_path)