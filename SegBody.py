import numpy as np
from PIL import Image
from transformers import pipeline

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_body(original_img, exclude_face=True, neck_inclusion=0.7):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create mask
    full_mask = np.zeros((img.height, img.width), dtype=np.uint8)
    face_mask = np.zeros((img.height, img.width), dtype=np.uint8)

    for s in segments:
        if s['label'] in ["Upper-clothes", "Pants", "Dress", "Skirt", "Left-arm", "Right-arm", "Left-leg", "Right-leg"]:
            full_mask = np.maximum(full_mask, s['mask'])
        elif s['label'] in ["Face", "Hair"]:
            face_mask = np.maximum(face_mask, s['mask'])

    if exclude_face:
        # Find the top and bottom of the face
        face_top = np.where(face_mask.sum(axis=1) > 0)[0][0]
        face_bottom = np.where(face_mask.sum(axis=1) > 0)[0][-1]
        
        # Calculate neck region
        neck_height = int((face_bottom - face_top) * neck_inclusion)
        neck_top = max(0, face_bottom - neck_height)
        
        # Create a mask that includes the body and the desired portion of the neck
        neck_mask = np.zeros_like(face_mask)
        neck_mask[neck_top:, :] = 1
        
        final_mask = full_mask | (face_mask & neck_mask)
    else:
        final_mask = full_mask

    # Ensure the mask covers from neck down
    rows_with_content = np.where(final_mask.sum(axis=1) > 0)[0]
    if len(rows_with_content) > 0:
        top_row = rows_with_content[0]
        final_mask[top_row:, :] = np.maximum(final_mask[top_row:, :], full_mask[top_row:, :])

    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Apply mask to original image
    img_rgba = img.convert('RGBA')
    img_rgba.putalpha(final_mask)

    return img_rgba, final_mask

def segment_torso(original_img):
    # This function remains unchanged
    ...

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