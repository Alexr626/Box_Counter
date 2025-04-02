#!/bin/bash

python train.py \
  --model_name shelf_mono_stereo_finetune \
  --split shelf \
  --dataset shelf \
  --data_path /drive2/jinboliu/csci677/images/original_images \
  --log_dir ./finetune_logs \
  --height 192 \
  --width 640 \
  --load_weights_folder ./models/mono+stereo_640x192 \
  --frame_ids 0 -1 1 \
  --batch_size 4 \
  --num_epochs 5