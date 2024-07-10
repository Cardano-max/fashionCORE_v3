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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Set up environment variables for sharing data
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

app = FastAPI()

class GenerationID(BaseModel):
    id: str

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

@app.post("/upload-images/")
async def upload_images(garment: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        garment_image = Image.open(io.BytesIO(await garment.read()))
        person_image = Image.open(io.BytesIO(await person.read()))

        garment_np = np.array(garment_image)
        person_np = np.array(person_image)

        generation_id = str(uuid.uuid4())
        
        # Store the images temporarily
        os.makedirs("temp_images", exist_ok=True)
        garment_path = f"temp_images/{generation_id}_garment.png"
        person_path = f"temp_images/{generation_id}_person.png"
        
        garment_image.save(garment_path)
        person_image.save(person_path)

        return JSONResponse(content={"generation_id": generation_id, "message": "Images uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/generation-id/{generation_id}")
async def get_generation_status(generation_id: str):
    garment_path = f"temp_images/{generation_id}_garment.png"
    person_path = f"temp_images/{generation_id}_person.png"
    
    if not (os.path.exists(garment_path) and os.path.exists(person_path)):
        raise HTTPException(status_code=404, detail="Generation ID not found")
    
    return JSONResponse(content={"status": "ready", "generation_id": generation_id})

@app.post("/virtual-try-on/{generation_id}")
async def run_virtual_try_on(generation_id: str):
    garment_path = f"temp_images/{generation_id}_garment.png"
    person_path = f"temp_images/{generation_id}_person.png"
    
    if not (os.path.exists(garment_path) and os.path.exists(person_path)):
        raise HTTPException(status_code=404, detail="Generation ID not found")

    try:
        garment_image = np.array(Image.open(garment_path))
        person_image = np.array(Image.open(person_path))

        result = virtual_try_on(garment_image, person_image)

        if result['success']:
            return JSONResponse(content={
                "success": True,
                "image_path": result['image_path'],
                "masked_image_path": result['masked_image_path']
            })
        else:
            raise HTTPException(status_code=500, detail=result['error'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result-image/{image_type}/{generation_id}")
async def get_result_image(image_type: str, generation_id: str):
    if image_type not in ["try-on", "masked"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = f"outputs/{generation_id}_{image_type}.png"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)