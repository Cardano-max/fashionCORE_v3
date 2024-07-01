import numpy as np
from PIL import Image
import rembg
import traceback

def rembg_run(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        output = rembg.remove(image)
        return np.array(output)
    except Exception as e:
        print("Error in rembg_run:", str(e))
        traceback.print_exc()
        return None
