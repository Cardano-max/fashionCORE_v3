import time
import traceback
import sys
import os
import numpy as np
import modules.config
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
from modules.util import HWC3, resize_image
from modules.masking import mask_clothes
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import io
import random

# Set up environment variables for sharing data
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

def virtual_try_on(clothes_image, person_image):
    try:
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)

        target_size = (1152, 896)
        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])

        # Generate mask using the mask_clothes function
        person_image_pil = Image.fromarray(person_image)
        inpaint_mask = mask_clothes(person_image_pil)

        # Display and save the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(inpaint_mask, cmap='gray')
        plt.axis('off')
        
        # Save the plot to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Generate a unique ID for this try-on session
        session_id = str(uuid.uuid4())

        # Save the mask image
        masked_image_path = os.path.join(modules.config.path_outputs, f"masked_image_{session_id}.png")
        with open(masked_image_path, 'wb') as f:
            f.write(buf.getvalue())
        
        plt.close()  # Close the plot to free up memory

        os.environ['MASKED_IMAGE_PATH'] = masked_image_path

        loras = []
        for lora in modules.config.default_loras:
            loras.extend(lora)

        args = [
            True,
            "",
            modules.config.default_prompt_negative,
            False,
            modules.config.default_styles,
            modules.config.default_performance,
            modules.config.default_aspect_ratio,
            1,
            modules.config.default_output_format,
            random.randint(constants.MIN_SEED, constants.MAX_SEED),
            modules.config.default_sample_sharpness,
            modules.config.default_cfg_scale,
            modules.config.default_base_model_name,
            modules.config.default_refiner_model_name,
            modules.config.default_refiner_switch,
        ] + loras + [
            True,
            "inpaint",
            flags.disabled,
            None,
            [],
            {'image': person_image, 'mask': inpaint_mask},
            "Wearing a new garment",
            inpaint_mask,
            True,
            True,
            modules.config.default_black_out_nsfw,
            1.5,
            0.8,
            0.3,
            modules.config.default_cfg_tsnr,
            modules.config.default_sampler,
            modules.config.default_scheduler,
            -1,
            -1,
            -1,
            -1,
            -1,
            modules.config.default_overwrite_upscale,
            False,
            True,
            False,
            False,
            100,
            200,
            flags.refiner_swap_method,
            0.5,
            False,
            1.0,
            1.0,
            1.0,
            1.0,
            False,
            False,
            modules.config.default_inpaint_engine_version,
            1.0,
            0.618,
            False,
            False,
            0,
            modules.config.default_save_metadata_to_images,
            modules.config.default_metadata_scheme,
        ]

        args.extend([
            clothes_image,
            0.86,
            0.97,
            flags.default_ip,
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            # Rename the output file to include the session ID
            output_path = os.path.join(modules.config.path_outputs, f"try_on_{session_id}.png")
            os.rename(task.results[0], output_path)
            return {"success": True, "image_path": output_path, "masked_image_path": masked_image_path}
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    from fast_api_wrapper import app
    
    uvicorn.run(app, host="0.0.0.0", port=8000)