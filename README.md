# Box_Counter Development Guidelines

## Project Overview
Box_Counter is a computer vision project that uses drone imagery and depth estimation to count boxes in storage bins. The workflow involves:
1. Preprocessing drone images of bins
2. Estimating depth maps using pre-trained models
3. Calibrating depth maps using known barcode positions
4. Constructing point clouds to count boxes

## Build & Run Commands
- Setup environment: `python -m venv venv && source venv/bin/activate && pip install -r venv_requirements.txt`
- Run complete pipeline: `python src/main.py`
- Run with specific depth model: `python src/main.py --depth_model="depth_anything"`
- Run individual components:
  - Image cropping: `python src/preprocessing/crop_images.py`
  - Image registration: `python src/preprocessing/register_bin_images.py`
  - Depth estimation: `python src/depth_map_estimation/depth_estimations.py`
  - Depth calibration: `python src/depth_map_estimation/extract_pixel_depths.py`
  
## Environment Setup & Run for VLM-based Box Counting
- Create virtual environment
  ```
  mamba create -n vlm_env python=3.10
  mamba activate vlm_env
  pip install -r requirements_vlm.txt
  ```
- Count boxes with ChatGPT API
  ```
  python -m src.count_boxes --model gpt --cache true
  ```
- Count boxes with local VLM
  ```
  python -m src.count_boxes --model Qwen/Qwen2.5-VL-7B-Instruct
  ```
- Fine-tune VLM for box counting
  ```
  python -m src.fine_tune 
  ```

## Code Style Guidelines

### Import Convention
```python
# Standard library imports first
import os
import sys
# Third-party imports next
import numpy as np
import torch
import cv2
# Local module imports last
from src.depth_map_estimation.utils import process_image
```

### Documentation
- Use docstrings with parameter descriptions
- Document function purpose, parameters, and return values

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`

### Error Handling
- Use descriptive error messages
- Handle exceptions with try/except blocks
- Return success/failure indicators from functions

### Type Hints
- Add type hints to function signatures where possible:
```python
def process_image(image_path: str, output_path: str) -> bool:
    """Process an image and return success state"""
```

### Data Management
- Store data files in the `data/` directory
- Use relative paths for data access
- Document data format in function docstrings

## Project Structure
- `src/main.py` - Main pipeline orchestration
- `src/preprocessing/` - Image preprocessing modules
  - `preprocess_raw_data.py` - Creates train/test splits and groups images by bin
  - `crop_images.py` - Removes borders from wide-angle camera images
  - `register_bin_images.py` - Aligns multiple images of the same bin
- `src/depth_map_estimation/` - Depth estimation and processing
  - `depth_estimations.py` - Generates depth maps using MiDaS or Depth Anything models
  - `extract_pixel_depths.py` - Calibrates depth maps using barcode positions
  - `point_cloud_construction.py` - Template for 3D point cloud creation (in development)
- `data/` - Contains all data files
  - `images/` - Various image directories (original, processed, cropped)
  - `depth/` - Contains depth maps (.pfm files) and calibration parameters (json)