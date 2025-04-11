import cv2
import numpy as np
from PIL import Image

def create_image_mask(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to identify black regions
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Optional: Clean up the mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # # Save the mask as an image
    # cv2.imwrite("test.jpg", mask)
    
    # # Optional: Create a colored visualization where mask is applied to original image
    # colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
    # masked_img = cv2.bitwise_and(img, colored_mask)
    # cv2.imwrite('masked_original.jpg', masked_img)
    
    
    return mask

if __name__=="__main__":
    mask_test = create_image_mask("data/images/examples/many_boxes/04-0207-02/1738952451758_e41cbabd-57e9-444f-868b-813e5c467612.png")
