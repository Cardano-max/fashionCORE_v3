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
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import io
import random
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import cv2
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch
from datetime import datetime, timedelta

app = FastAPI()

# Create a directory for temporary files
TEMP_DIR = "temp_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)

# Serve the temp directory
app.mount("/static", StaticFiles(directory=TEMP_DIR), name="static")

# Set up environment variables for sharing data
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

# Initialize Segformer model and processor
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

# Global variables for managing the queue and cooldown
is_processing = False
last_generation_time = None
cooldown_period = timedelta(minutes=5)

class GenerationID(BaseModel):
    id: str

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

def generate_mask(image):
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
    labels = [4, 14, 15]  # Upper Clothes 4, Left Arm 14, Right Arm 15

    combined_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in labels:
        combined_mask = torch.logical_or(combined_mask, pred_seg == label)

    pred_seg_new = torch.zeros_like(pred_seg)
    pred_seg_new[combined_mask] = 255

    image_mask = pred_seg_new.numpy().astype(np.uint8)

    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(image_mask, kernel, iterations=1)

    return dilated_mask

def virtual_try_on(clothes_image, person_image):
    try:
        # Convert person_image to PIL Image
        person_pil = Image.fromarray(person_image)

        # Generate mask
        inpaint_mask = generate_mask(person_pil)

        # Resize images and mask
        target_size = (512, 512)
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)
        inpaint_mask = HWC3(inpaint_mask)[:, :, 0]

        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])
        inpaint_mask = resize_image(inpaint_mask, target_size[0], target_size[1])

        # Save the mask image
        session_id = str(uuid.uuid4())
        masked_image_path = os.path.join(TEMP_DIR, f"masked_image_{session_id}.png")
        plt.imsave(masked_image_path, inpaint_mask, cmap='gray')

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
            output_path = os.path.join(TEMP_DIR, f"try_on_{session_id}.png")
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
async def run_virtual_try_on(generation_id: str, background_tasks: BackgroundTasks, request: Request):
    global is_processing, last_generation_time

    if is_processing:
        return JSONResponse(content={"message": "Another generation is in progress. Please try again later."})

    current_time = datetime.now()
    if last_generation_time and (current_time - last_generation_time) < cooldown_period:
        time_left = cooldown_period - (current_time - last_generation_time)
        return JSONResponse(content={"message": f"Please wait {time_left.seconds} seconds before starting a new generation."})

    garment_path = f"temp_images/{generation_id}_garment.png"
    person_path = f"temp_images/{generation_id}_person.png"
    
    if not (os.path.exists(garment_path) and os.path.exists(person_path)):
        raise HTTPException(status_code=404, detail="Generation ID not found")

    try:
        is_processing = True
        garment_image = np.array(Image.open(garment_path))
        person_image = np.array(Image.open(person_path))

        result = virtual_try_on(garment_image, person_image)

        if result['success']:
            last_generation_time = datetime.now()
            background_tasks.add_task(clean_up_files, generation_id)
            
            try_on_url = request.url_for('static', path=os.path.basename(result['image_path']))
            masked_url = request.url_for('static', path=os.path.basename(result['masked_image_path']))
            
            return JSONResponse(content={
                "success": True,
                "try_on_image_url": str(try_on_url),
                "masked_image_url": str(masked_url),
                "message": "Next generation will be enabled after 5 minutes"
            })
        else:
            raise HTTPException(status_code=500, detail=result['error'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        is_processing = False

@app.get("/result-image/{image_type}/{filename}")
async def get_result_image(image_type: str, filename: str):
    if image_type not in ["try-on", "masked"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = os.path.join(TEMP_DIR, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)

def clean_up_files(generation_id: str):
    garment_path = f"temp_images/{generation_id}_garment.png"
    person_path = f"temp_images/{generation_id}_person.png"
    
    if os.path.exists(garment_path):
        os.remove(garment_path)
    if os.path.exists(person_path):
        os.remove(person_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)