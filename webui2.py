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
from modules.masking import mask_clothes
import json
import torch
from PIL import Image

# Set up environment variables for sharing data
os.environ['GRADIO_PUBLIC_URL'] = ''
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
        inpaint_mask = np.array(inpaint_mask)

        # Save the masked image
        masked_image = person_image.copy()
        masked_image[inpaint_mask == 255] = [255, 0, 0]  # Highlight masked area in red
        masked_image_path = os.path.join(modules.config.path_outputs, f"masked_image_{int(time.time())}.png")
        Image.fromarray(masked_image).save(masked_image_path)
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
]

css = """
... (your existing CSS)
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #fff;
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
    masked_output = gr.Image(label="Masked Image", visible=False)
    try_on_output = gr.Image(label="Virtual Try-On Result", visible=False)
    image_link = gr.HTML(visible=True, elem_classes="result-links")
    error_output = gr.Textbox(label="Error", visible=False)

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if clothes_image is None or person_image is None:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Please upload both a garment image and a person image.", visible=True)
        
        # Show loading indicator
        yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        result = virtual_try_on(clothes_image, person_image)
        
        if result['success']:
            # Wait for the generated_image_path to be captured
            timeout = 30  # seconds
            start_time = time.time()
            while not os.environ['GENERATED_IMAGE_PATH']:
                if time.time() - start_time > timeout:
                    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Timeout waiting for image generation.", visible=True)
                    return
                time.sleep(0.5)
            
            generated_image_path = os.environ['GENERATED_IMAGE_PATH']
            masked_image_path = os.environ['MASKED_IMAGE_PATH']
            gradio_url = os.environ['GRADIO_PUBLIC_URL']
            
            if gradio_url and generated_image_path and masked_image_path:
                output_image_link = f"{gradio_url}/file={generated_image_path}"
                masked_image_link = f"{gradio_url}/file={masked_image_path}"
                link_html = f'<a href="{output_image_link}" target="_blank">View generated image</a> | <a href="{masked_image_link}" target="_blank">View masked image</a>'
                
                # Hide loading indicator and show the results
                yield gr.update(visible=False), gr.update(value=masked_image_path, visible=True), gr.update(value=generated_image_path, visible=True), gr.update(value=link_html, visible=True), gr.update(visible=False)
            else:
                yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=f"Unable to generate public links. Local file paths: Generated: {generated_image_path}, Masked: {masked_image_path}", visible=True), gr.update(visible=False)
        else:
            yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=result['error'], visible=True)

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[loading_indicator, masked_output, try_on_output, image_link, error_output]
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
    # Launch the Gradio app
    app, local_url, share_url = demo.launch(share=True, prevent_thread_lock=True)
    
    # Capture and print the public URL
    if share_url:
        os.environ['GRADIO_PUBLIC_URL'] = share_url
        print(f"Running on public URL: {share_url}")
    
    return app, local_url, share_url

# Launch the app using our custom function
custom_launch()

# Keep the script running
while True:
    time.sleep(1)