from rembg import remove
import gradio as gr
from PIL import Image
import numpy as np
import cv2

def rembg_run(input_image, progress=gr.Progress(track_tqdm=True)):
    try:
        if isinstance(input_image, np.ndarray):
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(input_image)
        elif isinstance(input_image, str):
            input_image = Image.open(input_image)
        
        output = remove(input_image)
        return np.array(output)
    except Exception as e:
        print(f"Error in rembg_run: {str(e)}")
        return input_image  # Return original image if removal fails