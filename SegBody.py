from transformers import pipeline
import numpy as np
from PIL import Image

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_body(original_img, exclude_face=True, neck_inclusion=0.5):
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

    if exclude_face:
        # Find the bottom of the face mask
        face_bottom = np.where(face_mask.sum(axis=1) > 0)[0][-1] if face_mask.sum() > 0 else 0
        
        # Calculate how much of the neck to include
        neck_height = int((img.height - face_bottom) * neck_inclusion)
        
        # Create a mask that includes the body and the desired portion of the neck
        neck_mask = np.zeros_like(face_mask)
        neck_mask[face_bottom:face_bottom + neck_height, :] = 1
        
        final_mask = body_mask | (face_mask & neck_mask)
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