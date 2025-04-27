import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def get_depth_estimations(model_id, model_name):
    # 1. Load model & image
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    cropped_image_dir = "data/images/cropped_images"
    output_dir = os.path.join("data/depth", model_name)
    os.makedirs(output_dir, exist_ok=True)

    predictions = []

    for root, _, files in os.walk(cropped_image_dir):
        for filepath in files:
            filename = filepath.rsplit('.', 1)[0]
            pfm_filepath = os.path.join(output_dir, f"{filename}-{model_name}.pfm")
            if os.path.exists(pfm_filepath):
                print("Continuing")
                continue
            
            image_path = os.path.join(root, filepath)
            image = Image.open(image_path).convert("RGB")
            inputs = image_processor(images=image, return_tensors="pt")

            # 2. Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            predictions.append(output)

            # print(output)

            Image.fromarray(output).save(pfm_filepath)
            # formatted = (output * 255 / np.max(output)).astype("uint8")
            # depth = Image.fromarray(formatted)
            # depth.show()
