from transformers import pipeline
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

# Initialize face detection
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def remove_face(img, mask):
    # Convert image to numpy array
    img_arr = np.asarray(img)
    
    # Run face detection
    faces = app.get(img_arr)
    
    if len(faces) == 0:
        return mask  # Return original mask if no face is detected
    
    # Get the first face
    face = faces[0]
    bbox = face.bbox

    # Width and height of face
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Make face locations bigger
    face_locations = [
        (bbox[0] - (w*0.5), bbox[1] - (h*0.5)),  # (x_left, y_top)
        (bbox[2] + (w*0.5), bbox[3] + (h*0.2))   # (x_right, y_bottom)
    ]

    # Draw black rect onto mask
    img1 = ImageDraw.Draw(mask)
    img1.rectangle(face_locations, fill=0)

    return mask

def segment_body(original_img, face=True):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag","Scarf"]
    mask_list = []
    for s in segments:
        if(s['label'] in segment_include):
            mask_list.append(s['mask'])

    # Paste all masks on top of each other 
    final_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        final_mask = np.maximum(final_mask, mask)
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Remove face
    if not face:
        final_mask = remove_face(img.convert('RGB'), final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img, final_mask

def segment_torso(original_img):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"]
    mask_list = []
    for s in segments:
        if(s['label'] in segment_include):
            mask_list.append(s['mask'])

    # Paste all masks on top of each other 
    final_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        final_mask = np.maximum(final_mask, mask)
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask * 255)

    # Remove face
    final_mask = remove_face(img.convert('RGB'), final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img, final_mask