import gradio as gr
import random
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
from modules.private_logger import get_current_html_path
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from modules.flags import Performance
from queue import Queue
from threading import Lock, Event

# Set up environment variables for sharing data
os.environ['GRADIO_PUBLIC_URL'] = ''
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

# Initialize Segformer model and processor
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

# Initialize queue and lock
task_queue = Queue()
queue_lock = Lock()
current_task_event = Event()

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
    labels = [4, 14, 15, 6, 12, 13]  # Upper Clothes 4, Left Arm 14, Right Arm 15

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
        target_size = (1024, 1024)
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)
        inpaint_mask = HWC3(inpaint_mask)[:, :, 0]

        # Use the original resize_image function from modules.util
        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])
        inpaint_mask = resize_image(inpaint_mask, target_size[0], target_size[1])

        # Display and save the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(inpaint_mask, cmap='gray')
        plt.axis('off')
        
        # Save the plot to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Save the mask image
        masked_image_path = os.path.join(modules.config.path_outputs, f"masked_image_{int(time.time())}.png")
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
            Performance.QUALITY.value,
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
            return {"success": True, "image_path": task.results[0], "masked_image_path": masked_image_path}
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}

example_garments = [
    "images/1.png",
    "images/2.png",
    "images/3.png",
    "images/4.png",
    "images/5.png",
    "images/6.png",
    "images/7.png",
    "images/8.png",
    "images/9.png",
    "images/10.png",
    "images/11.png",
    "images/12.jpeg",
    "images/13.jpeg",
    "images/14.jpeg",
    "images/15.jpeg",
    "images/16.jpeg",
    "images/17.jpeg",
    "images/18.jpeg",
    "images/19.jpeg",
    "images/20.jpeg",
    "images/21.jpeg",
    "images/22.jpeg",
    "images/23.jpeg",
    "images/24.jpeg",
    "images/25.jpeg",
    "images/26.png",
    "images/27.png",
    "images/28.png",
]

css = """
    body, .gradio-container {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .header {
        background-color: #2c2c2c;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #b3b3b3;
    }
    .example-garments img {
        border-radius: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .example-garments img:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
    }
    .try-on-button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .try-on-button:hover {
        background-color: #45a049;
    }
    .queue-info {
        background-color: #2c2c2c;
        border: 1px solid #3a3a3a;
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
        font-size: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .error-message {
        background-color: #ff3860;
        border: 1px solid #ff1443;
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
        font-size: 16px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .result-links a {
        color: #3273dc;
        text-decoration: none;
        margin: 0 10px;
        transition: color 0.3s ease;
    }
    .result-links a:hover {
        color: #2366d1;
        text-decoration: underline;
    }
    .loading {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #ffffff;
        animation: spin 1s ease-in-out infinite;
        -webkit-animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin {
        to { -webkit-transform: rotate(360deg); }
    }
    @-webkit-keyframes spin {
        to { -webkit-transform: rotate(360deg); }
    }
"""

def process_queue():
    while True:
        task = task_queue.get()
        if task is None:
            break
        clothes_image, person_image, result_callback = task
        current_task_event.set()
        result = virtual_try_on(clothes_image, person_image)
        current_task_event.clear()
        result_callback(result)
        task_queue.task_done()

# Start the queue processing thread
import threading
queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <div class="header">
            <h1 class="title">ArbiTryOn</h1>
            <p class="subtitle">Experience Arbisoft's merchandise with our cutting-edge virtual try-on system!</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Choose a Garment")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=2, rows=2, label="Example Garments", elem_class="example-garments")
            clothes_input = gr.Image(label="Selected Garment", source="upload", type="numpy")

        with gr.Column(scale=3):
            gr.Markdown("### Upload Your Photo")
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy")

    try_on_button = gr.Button("Try It On!", elem_classes="try-on-button")
    loading_indicator = gr.HTML('<div class="loading"></div>', visible=False)
    queue_info = gr.HTML(visible=False, elem_classes="queue-info")
    masked_output = gr.Image(label="Masked Image", visible=False)
    try_on_output = gr.Image(label="Virtual Try-On Result", visible=False)
    image_link = gr.HTML(visible=True, elem_classes="result-links")
    error_output = gr.HTML(visible(False), elem_classes="error-message")

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if clothes_image is None or person_image is None:
            return {
                loading_indicator: gr.update(visible=False),
                queue_info: gr.update(visible=False),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                error_output: gr.update(value="<p>Please upload both a garment image and a person image.</p>", visible=True),
                image_link: gr.update(visible=False)
            }

        # Show loading indicator and queue info
        yield {
            loading_indicator: gr.update(visible=True),
            queue_info: gr.update(value="<p>Your request is being processed. Please wait...</p>", visible=True),
            masked_output: gr.update(visible=False),
            try_on_output: gr.update(visible=False),
            error_output: gr.update(visible=False),
            image_link: gr.update(visible=False)
        }

        def result_callback(result):
            nonlocal generation_done, generation_result
            generation_done = True
            generation_result = result

        with queue_lock:
            current_position = task_queue.qsize()
            task_queue.put((clothes_image, person_image, result_callback))

        generation_done = False
        generation_result = None

        while not generation_done:
            if current_task_event.is_set():
                yield {
                    queue_info: gr.update(value="<p>Your request is being processed. This may take a few minutes. Please wait...</p>", visible=True)
                }
            else:
                current_position = max(0, current_position - 1)
                if current_position > 0:
                    yield {
                        queue_info: gr.update(value=f"<p>Your request is in queue. Current position: {current_position}</p><p>Estimated wait time: {current_position * 2} minutes</p>", visible=True)
                    }
                else:
                    yield {
                        queue_info: gr.update(value="<p>Your request is next in line. Processing will begin shortly...</p>", visible=True)
                    }
            time.sleep(5)

        if generation_result is None:
            yield {
                loading_indicator: gr.update(visible=False),
                queue_info: gr.update(visible=False),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                image_link: gr.update(visible=False),
                error_output: gr.update(value="<p>An unexpected error occurred. Please try again later.</p>", visible=True)
            }
        elif generation_result['success']:
            generated_image_path = os.environ['GENERATED_IMAGE_PATH']
            masked_image_path = os.environ['MASKED_IMAGE_PATH']
            gradio_url = os.environ['GRADIO_PUBLIC_URL']

            if gradio_url and generated_image_path and masked_image_path:
                output_image_link = f"{gradio_url}/file={generated_image_path}"
                masked_image_link = f"{gradio_url}/file={masked_image_path}"
                link_html = f'<a href="{output_image_link}" target="_blank">View generated image</a> | <a href="{masked_image_link}" target="_blank">View masked image</a>'

                yield {
                    loading_indicator: gr.update(visible=False),
                    queue_info: gr.update(visible=False),
                    masked_output: gr.update(value=masked_image_path, visible=True),
                    try_on_output: gr.update(value=generated_image_path, visible=True),
                    image_link: gr.update(value=link_html, visible=True),
                    error_output: gr.update(visible=False)
                }
            else:
                yield {
                    loading_indicator: gr.update(visible=False),
                    queue_info: gr.update(visible=False),
                    masked_output: gr.update(visible=False),
                    try_on_output: gr.update(visible=False),
                    image_link: gr.update(visible(False)),
                    error_output: gr.update(value="<p>Unable to generate public links. Please try again later.</p>", visible=True)
                }
        else:
            yield {
                loading_indicator: gr.update(visible(False)),
                queue_info: gr.update(visible(False)),
                masked_output: gr.update(visible(False)),
                try_on_output: gr.update(visible(False)),
                image_link: gr.update(visible(False)),
                error_output: gr.update(value=f"<p>Error: {generation_result['error']}</p><p>Please try again later.</p>", visible(True))
            }

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[loading_indicator, queue_info, masked_output, try_on_output, image_link, error_output]
    )

    gr.Markdown(
        """
        ## How It Works
        1. Choose a garment from our examples or upload your own.
        2. Upload a photo of yourself.
        3. Click "Try It On!" to see the magic happen!

        Experience the future of online shopping with ArbiTryOn - where technology meets style!
        """
    )

demo.queue()

def custom_launch():
    app, local_url, share_url = demo.launch(share=True, prevent_thread_lock=True)
    
    if share_url:
        os.environ['GRADIO_PUBLIC_URL'] = share_url
        print(f"Running on public URL: {share_url}")
    
    return app, local_url, share_url

custom_launch()

# Keep the script running
while True:
    time.sleep(1)
