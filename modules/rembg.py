from rembg import remove
import gradio as gr
from PIL import Image
import numpy as np

def rembg_run(input_image, progress=gr.Progress(track_tqdm=True)):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    elif isinstance(input_image, str):
        input_image = Image.open(input_image)
    else:
        raise ValueError("Input must be either a numpy array or a file path")
    
    output = remove(input_image)
    return np.array(output)