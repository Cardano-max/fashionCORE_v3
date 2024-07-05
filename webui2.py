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

# Set up environment variables for sharing data
os.environ['GRADIO_PUBLIC_URL'] = ''
os.environ['GENERATED_IMAGE_PATH'] = ''

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

from extras.inpaint_mask import generate_mask_from_image
import traceback

def virtual_try_on(clothes_image, person_image, inpaint_mask):
    try:
        # Convert images to numpy arrays if they're not already
        clothes_image = np.array(clothes_image)
        person_image = np.array(person_image)
        inpaint_mask = np.array(inpaint_mask)

        # Prepare LoRA arguments
        loras = []
        for lora in modules.config.default_loras:
            loras.append(lora[0])
            loras.append(lora[1])

        # Set up the arguments for the generation task
        args = [
            True,  # generate_image_grid
            "",  # prompt (empty as per manual metadata)
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
            "",  # inpaint_additional_prompt
            inpaint_mask,  # inpaint_mask_image_upload
            False,  # disable_preview
            False,  # disable_intermediate_results
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
            return task.results[0]  # Return the first (and only) generated image
        else:
            return "Error: No results generated"

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        traceback.print_exc()
        return f"Error: {str(e)}"

        
example_garments = [
    "images/first.png",
    "images/second.png",
    "images/third.png",
    "images/first.png",
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
        with gr.Column(scale=1):
            gr.Markdown("### Choose a Garment")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=2, rows=2, label="Example Garments", elem_class="example-garments")
            clothes_input = gr.Image(label="Selected Garment", source="upload", type="numpy")

        with gr.Column(scale=1):
            gr.Markdown("### Upload Your Photo")
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy", tool="sketch", elem_id="inpaint_canvas")

    try_on_button = gr.Button("Try It On!", elem_classes="try-on-button")
    loading_indicator = gr.HTML('<div class="loading"></div>', visible=False)
    try_on_output = gr.Image(label="Virtual Try-On Result", visible=False)
    image_link = gr.HTML(visible=True, elem_classes="result-links")
    error_output = gr.Textbox(label="Error", visible=False)

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)


def process_virtual_try_on(clothes_image, person_image):
    if clothes_image is None or person_image is None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Please upload both a garment image and a person image.", visible=True)
    
    inpaint_image = person_image['image']
    inpaint_mask = person_image['mask']
    
    if inpaint_mask is None or np.sum(inpaint_mask) == 0:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Please draw a mask on the person image to indicate where to apply the garment.", visible=True)
    
    # Show loading indicator
    yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    result = virtual_try_on(clothes_image, inpaint_image, inpaint_mask)
    
    if isinstance(result, str) and result.startswith("Error"):
        yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=result, visible=True)
    else:
        # Wait for the generated_image_path to be captured
        timeout = 30  # seconds
        start_time = time.time()
        while not os.environ.get('GENERATED_IMAGE_PATH'):
            if time.time() - start_time > timeout:
                yield gr.update(visible=False), gr.update(visible=False), gr.update(value="Timeout waiting for image generation.", visible=True), gr.update(visible=False)
                return
            time.sleep(0.5)
        
        generated_image_path = os.environ['GENERATED_IMAGE_PATH']
        gradio_url = os.environ.get('GRADIO_PUBLIC_URL', '')
        
        if gradio_url and generated_image_path:
            output_image_link = f"{gradio_url}/file={generated_image_path}"
            link_html = f'<a href="{output_image_link}" target="_blank">Click here to view the generated image</a>'
            
            # Hide loading indicator and show the result
            yield gr.update(visible=False), gr.update(value=generated_image_path, visible=True), gr.update(value=link_html, visible=True), gr.update(visible=False)
        else:
            yield gr.update(visible=False), gr.update(visible=False), gr.update(value=f"Unable to generate public link. Local file path: {generated_image_path}", visible=True), gr.update(visible=False)

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[loading_indicator, try_on_output, image_link, error_output]
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