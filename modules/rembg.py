from rembg import remove
from PIL import Image
import numpy as np

def rembg_run(input_image):
    if isinstance(input_image, str):
        # If input is a file path
        with Image.open(input_image) as img:
            input_array = np.array(img)
    elif isinstance(input_image, np.ndarray):
        # If input is already a numpy array
        input_array = input_image
    else:
        raise ValueError(f"Input type {type(input_image)} is not supported.")
    
    output_array = remove(input_array)
    return Image.fromarray(output_array)