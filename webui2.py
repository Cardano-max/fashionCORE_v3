import gradio as gr
import random
import time
import json
import traceback
import sys
import os
import numpy as np
import modules.config
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
from modules.util import HWC3, resize_image
from modules.private_logger import log

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler


def virtual_try_on(clothes_image, person_image, inpaint_mask):
    try:
        # Convert images to numpy arrays if they're not already
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)
        inpaint_mask = HWC3(inpaint_mask)[:, :, 0]

        # Resize images to match
        target_size = (512, 512)  # You can adjust this size
        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])
        inpaint_mask = resize_image(inpaint_mask, target_size[0], target_size[1])

        # Prepare LoRA arguments
        loras = []
        for lora in modules.config.default_loras:
            loras.extend(lora)

        # Set up the arguments for the generation task
        args = [
            True,  # generate_image_grid
            "",  # prompt
            modules.config.default_prompt_negative,  # negative_prompt
            False,  # translate_prompts
            modules.config.default_styles,  # style_selections
            modules.config.default_performance,  # performance_selection
            modules.config.default_aspect_ratio,  # aspect_ratios_selection
            1,  # image_number
            modules.config.default_output_format,  # output_format
            random.randint(constants.MIN_SEED, constants.MAX_SEED),  # image_seed
            modules.config.default_sample_sharpness,  # sharpness
            modules.config.default_cfg_scale,  # guidance_scale
            modules.config.default_base_model_name,  # base_model_name
            modules.config.default_refiner_model_name,  # refiner_model_name
            modules.config.default_refiner_switch,  # refiner_switch
        ] + loras + [
            True,  # input_image_checkbox
            "inpaint",  # current_tab
            flags.disabled,  # uov_method
            None,  # uov_input_image
            [],  # outpaint_selections
            {'image': person_image, 'mask': inpaint_mask},  # inpaint_input_image
            "Wearing a new garment",  # inpaint_additional_prompt
            inpaint_mask,  # inpaint_mask_image_upload
            True,  # disable_preview
            True,  # disable_intermediate_results
            modules.config.default_black_out_nsfw,  # black_out_nsfw
            1.5,  # adm_scaler_positive
            0.8,  # adm_scaler_negative
            0.3,  # adm_scaler_end
            modules.config.default_cfg_tsnr,  # adaptive_cfg
            modules.config.default_sampler,  # sampler_name
            modules.config.default_scheduler,  # scheduler_name
            -1,  # overwrite_step
            -1,  # overwrite_switch
            -1,  # overwrite_width
            -1,  # overwrite_height
            -1,  # overwrite_vary_strength
            modules.config.default_overwrite_upscale,  # overwrite_upscale_strength
            False,  # mixing_image_prompt_and_vary_upscale
            True,  # mixing_image_prompt_and_inpaint
            False,  # debugging_cn_preprocessor
            False,  # skipping_cn_preprocessor
            100,  # canny_low_threshold
            200,  # canny_high_threshold
            flags.refiner_swap_method,  # refiner_swap_method
            0.5,  # controlnet_softness
            False,  # freeu_enabled
            1.0,  # freeu_b1
            1.0,  # freeu_b2
            1.0,  # freeu_s1
            1.0,  # freeu_s2
            False,  # debugging_inpaint_preprocessor
            False,  # inpaint_disable_initial_latent
            modules.config.default_inpaint_engine_version,  # inpaint_engine
            1.0,  # inpaint_strength
            0.618,  # inpaint_respective_field
            True,  # inpaint_mask_upload_checkbox
            False,  # invert_mask_checkbox
            0,  # inpaint_erode_or_dilate
            modules.config.default_save_metadata_to_images,  # save_metadata_to_images
            modules.config.default_metadata_scheme,  # metadata_scheme
        ]

        # Add Image Prompt for clothes image
        args.extend([
            clothes_image,  # ip_image
            0.6,  # ip_stop
            0.5,  # ip_weight
            flags.default_ip,  # ip_type
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        # Wait for the task to complete
        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if isinstance(task.results, list) and len(task.results) > 0:
            return json.dumps({"success": True, "image_path": task.results[0]})
        else:
            return json.dumps({"success": False, "error": "No results generated"})

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        traceback.print_exc()
        async_task.yields.append(['error', str(e)])
    finally:
        async_task.processing = False

# Example garment images (replace with actual image paths)
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
    error_output = gr.Textbox(label="Error", visible=False)

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if clothes_image is None or person_image is None:
            return gr.update(value=None, visible=False), gr.update(value="Please upload both a garment image and a person image.", visible=True)
        
        inpaint_image = person_image['image']
        inpaint_mask = person_image['mask']
        
        if inpaint_mask is None or np.sum(inpaint_mask) == 0:
            return gr.update(value=None, visible=False), gr.update(value="Please draw a mask on the person image to indicate where to apply the garment.", visible=True)
        
        result = virtual_try_on(clothes_image, inpaint_image, inpaint_mask)
        try:
            result_json = json.loads(result)
            if result_json['success']:
                return gr.update(value=result_json['image_path'], visible=True), gr.update(value="", visible=False)
            else:
                return gr.update(value=None, visible=False), gr.update(value=result_json['error'], visible=True)
        except json.JSONDecodeError:
            return gr.update(value=None, visible=False), gr.update(value="Server returned an invalid response.", visible=True)

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[try_on_output, error_output]
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
