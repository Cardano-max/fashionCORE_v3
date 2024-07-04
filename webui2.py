import gradio as gr
import random
import time
import modules.config
import modules.flags as flags
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import numpy as np
from PIL import Image

# Function to load example garment images
def load_example_garments():
    # Replace these with actual paths to your garment images
    return [
        "images/first.png",
        "images/second.png",
        "images/third.png",
        "images/first.png",
        "images/first.png",
    ]

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

# Create the Gradio interface
css = """
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
h1 {
    color: #333;
    text-align: center;
}
.description {
    text-align: center;
    margin-bottom: 20px;
    font-size: 18px;
    color: #666;
}
.example-garments {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
}
.example-garment {
    cursor: pointer;
    transition: transform 0.2s;
}
.example-garment:hover {
    transform: scale(1.05);
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<div class='container'>")
    gr.HTML("<h1>ArbiTryOn - Virtual Fitting Room</h1>")
    gr.HTML("<p class='description'>Experience Arbisoft's merchandise like never before! Upload your photo, choose or upload a garment, and see how it looks on you instantly.</p>")
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<h3>Step 1: Choose a Garment</h3>")
            example_garments = gr.Dataset(
                components=[gr.Image(type="numpy", label="Selected Garment")],
                samples=[[img] for img in load_example_garments()],
                type="index",
                label="Example Garments"
            )
            clothes_input = gr.Image(label="Garment Image", source='upload', type='numpy')
        
        with gr.Column():
            gr.HTML("<h3>Step 2: Upload Your Photo</h3>")
            person_input = gr.Image(label="Your Photo", source='upload', type='numpy', tool='sketch', elem_id='inpaint_canvas')
    
    with gr.Row():
        try_on_button = gr.Button("Try It On!", variant="primary")
    
    with gr.Row():
        try_on_output = gr.Image(label="See How It Looks!")
    
    error_output = gr.Textbox(label="Error", visible=False)
    
    gr.HTML("</div>")

    def update_garment(evt: gr.SelectData):
        return evt.value[0]
    
    example_garments.select(update_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if person_image is None:
            return None, "Please upload your photo."
        if clothes_image is None:
            return None, "Please select or upload a garment."
        
        inpaint_image = person_image['image']
        inpaint_mask = person_image['mask']
        
        result = virtual_try_on(clothes_image, inpaint_image, inpaint_mask)
        if isinstance(result, str):  # Error occurred
            return None, result
        else:  # Successfully generated image
            return result, ""

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[try_on_output, error_output]
    )

demo.launch(share=True)