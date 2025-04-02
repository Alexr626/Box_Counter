#!/bin/bash


# The weight directory after fine-tuning(encoder.pth depth.pth):
LOAD_FOLDER="./finetune_logs/shelf_mono_stereo_finetune/models/weights_4"

# Specify the path to a single image or folder to be tested:
# e.g. IMG_OR_DIR="./my_images/1738952395743_6323dced-fe24-474e-a63a-666bd674e91d.jpg"
IMG_OR_DIR="./assets/1738952464426_e6e587a4-4c05-4065-ab21-a5e3263cdeb9.jpg"

python test_simple.py \
  --image_path $IMG_OR_DIR \
  --model_name mono+stereo_640x192 \
  --load_weights_folder $LOAD_FOLDER \
  --pred_metric_depth
