from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import uuid
import os
import time
from webui_api import virtual_try_on

app = FastAPI()

class GenerationID(BaseModel):
    id: str

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