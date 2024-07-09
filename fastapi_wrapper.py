from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import uuid
from PIL import Image
import io
import numpy as np

# Import your virtual_try_on function
from webui2 import virtual_try_on

app = FastAPI()

# In-memory storage for job status
job_status = {}

@app.post("/upload")
async def upload_images(clothes: UploadFile = File(...), person: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
        # Read and convert images to numpy arrays
        clothes_image = np.array(Image.open(io.BytesIO(await clothes.read())))
        person_image = np.array(Image.open(io.BytesIO(await person.read())))
        mask_image = np.array(Image.open(io.BytesIO(await mask.read())))
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Store images and update job status
        job_status[job_id] = {
            "status": "uploaded",
            "clothes": clothes_image,
            "person": person_image,
            "mask": mask_image
        }
        
        return {"job_id": job_id, "status": "Images uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process/{job_id}")
async def process_images(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    if job["status"] != "uploaded":
        return {"job_id": job_id, "status": job["status"]}
    
    # Update job status
    job["status"] = "processing"
    
    # Run the virtual try-on process asynchronously
    asyncio.create_task(run_virtual_try_on(job_id))
    
    return {"job_id": job_id, "status": "Processing started"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    return {"job_id": job_id, "status": job["status"]}

async def run_virtual_try_on(job_id):
    job = job_status[job_id]
    try:
        result = virtual_try_on(job["clothes"], job["person"], job["mask"])
        if result["success"]:
            job["status"] = "completed"
            job["result"] = result["image_path"]
        else:
            job["status"] = "failed"
            job["error"] = result["error"]
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)