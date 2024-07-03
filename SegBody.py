from transformers import pipeline
import numpy as np
from PIL import Image

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_body(original_img, exclude_face=True):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Left-leg", "Right-leg", "Left-arm", "Right-arm"]
    face_segments = ["Hat", "Hair", "Sunglasses", "Face"]
    
    body_mask = np.zeros((img.height, img.width), dtype=np.uint8)
    face_mask = np.zeros((img.height, img.width), dtype=np.uint8)

    for s in segments:
        if s['label'] in segment_include:
            body_mask = np.maximum(body_mask, s['mask'])
        elif s['label'] in face_segments:
            face_mask = np.maximum(face_mask, s['mask'])

    # If we want to exclude the face, subtract the face mask from the body mask
    if exclude_face:
        final_mask = body_mask & ~face_mask
    else:
        final_mask = body_mask

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
    segment_include = ["Upper-clothes", "Dress", "Belt", "Left-arm", "Right-arm"]
    
    final_mask = np.zeros((img.height, img.width), dtype=np.uint8)

    for s in segments:
        if s['label'] in segment_include:
            final_mask = np.maximum(final_mask, s['mask'])

    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Apply mask to original image
    img_rgba = img.convert('RGBA')
    img_rgba.putalpha(final_mask)

    return img_rgba, final_mask