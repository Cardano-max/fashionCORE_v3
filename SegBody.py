from transformers import pipeline
import numpy as np
from PIL import Image
import cv2

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_body(person_image, neck_inclusion=0.7, edge_smoothing=5, edge_expansion=3):
    # Segment person image
    segments = segmenter(person_image)
    
    # Create body mask
    body_mask = np.zeros((person_image.height, person_image.width), dtype=np.uint8)
    face_mask = np.zeros_like(body_mask)
    
    body_parts = ["Upper-clothes", "Pants", "Skirt", "Dress", "Belt", "Left-shoe", "Right-shoe", "Left-leg", "Right-leg", "Left-arm", "Right-arm"]
    face_parts = ["Face", "Hair"]
    
    for s in segments:
        if s['label'] in body_parts:
            body_mask = np.maximum(body_mask, s['mask'])
        elif s['label'] in face_parts:
            face_mask = np.maximum(face_mask, s['mask'])
    
    # Find the bottom of the face mask
    face_bottom = np.where(face_mask.sum(axis=1) > 0)[0][-1] if face_mask.sum() > 0 else 0
    
    # Calculate how much of the neck to include
    neck_height = int((person_image.height - face_bottom) * neck_inclusion)
    
    # Create a mask that includes the body and the desired portion of the neck
    neck_mask = np.zeros_like(face_mask)
    neck_mask[face_bottom:face_bottom + neck_height, :] = 1
    
    # Combine body mask with neck area
    final_mask = body_mask | neck_mask
    
    # Apply Gaussian blur to smooth edges
    final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (edge_smoothing, edge_smoothing), 0)
    
    # Threshold the blurred mask to create binary mask
    _, final_mask = cv2.threshold(final_mask, 0.5, 1, cv2.THRESH_BINARY)
    
    # Expand the mask edges
    kernel = np.ones((edge_expansion, edge_expansion), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    # Convert final mask to PIL Image
    final_mask = Image.fromarray((final_mask * 255).astype(np.uint8))
    
    # Apply mask to original image
    img_rgba = person_image.convert('RGBA')
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