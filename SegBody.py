from transformers import pipeline
import numpy as np
from PIL import Image

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_body(original_img, face=True):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
    mask_list = []
    for s in segments:
        if s['label'] in segment_include:
            mask_list.append(s['mask'])

    # Paste all masks on top of each other 
    final_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        final_mask = np.maximum(final_mask, mask)
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Apply mask to original image
    img_rgba = img.convert('RGBA')
    img_rgba.putalpha(final_mask)

    return img_rgba, final_mask

def segment_torso(original_img):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"]
    mask_list = []
    for s in segments:
        if s['label'] in segment_include:
            mask_list.append(s['mask'])

    # Paste all masks on top of each other 
    final_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        final_mask = np.maximum(final_mask, mask)
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Apply mask to original image
    img_rgba = img.convert('RGBA')
    img_rgba.putalpha(final_mask)

    return img_rgba, final_mask