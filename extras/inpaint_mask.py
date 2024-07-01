from PIL import Image
import numpy as np
import torch
from rembg import remove, new_session
from extras.GroundingDINO.util.inference import default_groundingdino

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_grounded_sam(input_image, text_prompt, box_threshold, text_threshold):

    # run grounding dino model
    boxes, _ = default_groundingdino(
        image=np.array(input_image),
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes.xyxy


def generate_mask_from_image(image, mask_model, extras):
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        boxes = run_grounded_sam(Image.fromarray(image), extras['sam_prompt_text'], box_threshold=extras['box_threshold'], text_threshold=extras['text_threshold'])
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]]) if len(boxes) == 0 else boxes
        extras['sam_prompt'] = []
        for idx, box in enumerate(boxes):
            extras['sam_prompt'] += [{"type": "rectangle", "data": box.tolist()}]

    mask = remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        **extras
    )

    # Ensure the mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    elif len(mask.shape) != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    return mask