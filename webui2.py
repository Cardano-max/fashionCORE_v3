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

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

def virtual_try_on(clothes_image, person_image, inpaint_mask):
    try:
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)
        inpaint_mask = HWC3(inpaint_mask)[:, :, 0]

        target_size = (512, 512)
        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])
        inpaint_mask = resize_image(inpaint_mask, target_size[0], target_size[1])

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
            True,
            False,
            0,
            modules.config.default_save_metadata_to_images,
            modules.config.default_metadata_scheme,
        ]

        args.extend([
            clothes_image,
            0.6,
            0.5,
            flags.default_ip,
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            return {"success": True, "image_path": task.results[0]}
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}

example_garments = [
    "images/first.png",
    "images/second.png",
    "images/third.png",
    "images/first.png",
]

css = """
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 30px;
}
.title {
    font-size: 36px;
    color: #333;
    margin-bottom: 10px;
}
.subtitle {
    font-size: 18px;
    color: #666;
}
.example-garments {
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
}
.example-garment {
    cursor: pointer;
    transition: transform 0.3s ease;
}
.example-garment:hover {
    transform: scale(1.05);
}
.try-on-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.try-on-button:hover {
    background-color: #45a049;
}
.result-links {
    margin-top: 20px;
    text-align: center;
}
.result-links a {
    display: inline-block;
    margin: 10px;
    padding: 10px 20px;
    background-color: #4CAF50;
    color: white;
    text-decoration: none;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
.result-links a:hover {
    background-color: #45a049;
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
        with gr.Column(scale=1):
            gr.Markdown("### Choose a Garment")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=2, rows=2, label="Example Garments", elem_class="example-garments")
            clothes_input = gr.Image(label="Selected Garment", source="upload", type="numpy")

        with gr.Column(scale=1):
            gr.Markdown("### Upload Your Photo")
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy", tool="sketch", elem_id="inpaint_canvas")

    try_on_button = gr.Button("Try It On!", elem_classes="try-on-button")
    try_on_output = gr.Image(label="Virtual Try-On Result")
    image_link = gr.HTML(visible=True, elem_classes="result-links")
    error_output = gr.Textbox(label="Error", visible=False)

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if clothes_image is None or person_image is None:
            return gr.update(value=None, visible=False), gr.update(visible=False), gr.update(value="Please upload both a garment image and a person image.", visible=True)
        
        inpaint_image = person_image['image']
        inpaint_mask = person_image['mask']
        
        if inpaint_mask is None or np.sum(inpaint_mask) == 0:
            return gr.update(value=None, visible=False), gr.update(visible=False), gr.update(value="Please draw a mask on the person image to indicate where to apply the garment.", visible=True)
        
        result = virtual_try_on(clothes_image, inpaint_image, inpaint_mask)
        
        if result['success']:
            image_path = result['image_path']
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, start=os.getcwd())
                history_log_path = get_current_html_path()
                link_html = f'<a href="{relative_path}" target="_blank">Click here to view the generated image</a><br>'
                link_html += f'<a href="file={history_log_path}" target="_blank">View History Log</a>'
                return gr.update(value=image_path, visible=True), gr.update(value=link_html, visible=True), gr.update(value="", visible=False)
            else:
                return gr.update(value=None, visible=False), gr.update(visible=False), gr.update(value=f"Generated image not found at {image_path}", visible=True)
        else:
            return gr.update(value=None, visible=False), gr.update(visible=False), gr.update(value=result['NO'], visible=True)

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[try_on_output, image_link, error_output]
    )

    gr.Markdown(
        """
        ## How It Works
        1. Choose a garment from our examples or upload your own.
        2. Upload a photo of yourself and use the brush tool to indicate where you want the garment placed.
        3. Click "Try It On!" to see the magic happen!

        Experience the future of online shopping with ArbiTryOn - where technology meets style!
        """
    )

demo.launch(share=True)