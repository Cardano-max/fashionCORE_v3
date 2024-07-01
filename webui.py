import gradio as gr
import numpy as np
from PIL import Image
from rembg import remove
from modules.rembg import rembg_run
from extras.inpaint_mask import generate_mask_from_image
import modules.flags as flags
from modules.async_worker import AsyncTask
import time

def virtual_tryon_pipeline(clothes_image, person_image):
    # Step 1: Remove background from clothes image
    clothes_no_bg = rembg_run(clothes_image)

    # Step 2: Generate mask for person image
    person_mask = generate_mask_from_image(
        person_image,
        'sam',
        {
            'sam_prompt_text': 'Clothes',
            'sam_model': 'sam_vit_b_01ec64',
            'sam_quant': False,
            'box_threshold': 0.3,
            'text_threshold': 0.25
        }
    )

    # Step 3: Prepare inputs for the main generation process
    task = AsyncTask([
        "",  # prompt
        "",  # negative_prompt
        False,  # translate_prompts
        ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],  # style_selections
        "Quality",  # performance_selection
        "1152×896",  # aspect_ratios_selection
        1,  # image_number
        "png",  # output_format
        -1,  # image_seed
        2.0,  # sharpness
        7.0,  # guidance_scale
        "model.safetensors",  # base_model_name
        "None",  # refiner_model_name
        0.8,  # refiner_switch
        [["None", 1.0]] * 5,  # loras
        True,  # input_image_checkbox
        "ip",  # current_tab
        "Disabled",  # uov_method
        None,  # uov_input_image
        [],  # outpaint_selections
        {'image': person_image, 'mask': person_mask},  # inpaint_input_image
        "",  # inpaint_additional_prompt
        None,  # inpaint_mask_image_upload
        False,  # disable_preview
        False,  # disable_intermediate_results
        False,  # black_out_nsfw
        1.0,  # adm_scaler_positive
        1.0,  # adm_scaler_negative
        0.0,  # adm_scaler_end
        1.0,  # adaptive_cfg
        "dpmpp_2m_sde_gpu",  # sampler_name
        "karras",  # scheduler_name
        -1,  # overwrite_step
        -1,  # overwrite_switch
        -1,  # overwrite_width
        -1,  # overwrite_height
        -1,  # overwrite_vary_strength
        -1,  # overwrite_upscale_strength
        True,  # mixing_image_prompt_and_vary_upscale
        True,  # mixing_image_prompt_and_inpaint
        False,  # debugging_cn_preprocessor
        False,  # skipping_cn_preprocessor
        100,  # canny_low_threshold
        200,  # canny_high_threshold
        "joint",  # refiner_swap_method
        0.5,  # controlnet_softness
        False,  # freeu_enabled
        1.0,  # freeu_b1
        1.0,  # freeu_b2
        1.0,  # freeu_s1
        1.0,  # freeu_s2
        False,  # debugging_inpaint_preprocessor
        False,  # inpaint_disable_initial_latent
        "v2.6",  # inpaint_engine
        1.0,  # inpaint_strength
        0.618,  # inpaint_respective_field
        True,  # inpaint_mask_upload_checkbox
        False,  # invert_mask_checkbox
        0,  # inpaint_erode_or_dilate
        clothes_no_bg,  # First image prompt (clothes without background)
        0.86,  # Stop at for clothes image
        0.97,  # Weight for clothes image
        flags.cn_ip,  # Type for clothes image
    ])

    # Step 4: Generate the image
    from webui import generate_clicked
    result = generate_clicked(task)

    # Wait for the task to complete
    while not task.processing:
        time.sleep(0.1)
    while task.processing:
        time.sleep(0.1)

    return task.results

# Add this to your Gradio interface
with gr.Blocks() as virtual_tryon_interface:
    with gr.Row():
        clothes_input = gr.Image(label="Clothes Image", type="numpy")
        person_input = gr.Image(label="Person Image", type="numpy")
    generate_button = gr.Button("Generate Virtual Try-On")
    output_gallery = gr.Gallery(label="Results")

    generate_button.click(
        virtual_tryon_pipeline,
        inputs=[clothes_input, person_input],
        outputs=[output_gallery]
    )

# Launch the interface
virtual_tryon_interface.launch(share=True)
