import torch
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

def mask_clothes(image, label=4):
    # image = Image.open("/content/images.jpeg")
    inputs = processor(images=image, return_tensors="pt")


    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # Create a mask where the values are 4 (the label for Upper Body Clothes)
    mask = pred_seg == 4

    # Create a new tensor filled with 0s 
    pred_seg_new = torch.zeros_like(pred_seg)

    # Set the values at the positions where the mask is True to 255 
    pred_seg_new[mask] = 255

    image_mask = pred_seg_new.numpy().astype(np.uint8)

    return image_mask