import numpy as np
from PIL import Image
from modules.rembg import rembg_run
from extras.inpaint_mask import generate_mask_from_image
import modules.flags as flags
from modules.async_worker import AsyncTask
import time
import modules.config
import modules.constants as constants
import random

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
    seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    task = AsyncTask([
        True,  # generate_image_grid
        "",  # prompt
        "",  # negative_prompt
        False,  # translate_prompts
        ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],  # style_selections
        flags.Performance.QUALITY.value,  # performance_selection
        "1152×896",  # aspect_ratios_selection
        1,  # image_number
        modules.config.default_output_format,  # output_format
        seed,  # image_seed
        modules.config.default_sample_sharpness,  # sharpness
        modules.config.default_cfg_scale,  # guidance_scale
        modules.config.default_base_model_name,  # base_model_name
        modules.config.default_refiner_model_name,  # refiner_model_name
        modules.config.default_refiner_switch,  # refiner_switch
        *[item for sublist in modules.config.default_loras for item in sublist],  # loras
        True,  # input_image_checkbox
        "ip",  # current_tab
        flags.disabled,  # uov_method
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
        modules.config.default_cfg_tsnr,  # adaptive_cfg
        modules.config.default_sampler,  # sampler_name
        modules.config.default_scheduler,  # scheduler_name
        modules.config.default_overwrite_step,  # overwrite_step
        modules.config.default_overwrite_switch,  # overwrite_switch
        -1,  # overwrite_width
        -1,  # overwrite_height
        -1,  # overwrite_vary_strength
        modules.config.default_overwrite_upscale,  # overwrite_upscale_strength
        True,  # mixing_image_prompt_and_vary_upscale
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
        False,  # inpaint_mask_upload_checkbox
        False,  # invert_mask_checkbox
        0,  # inpaint_erode_or_dilate
        clothes_no_bg,  # First image prompt (clothes without background)
        0.86,  # Stop at for clothes image
        0.97,  # Weight for clothes image
        flags.cn_ip,  # Type for clothes image
    ])

    # Step 4: Add the task to the async_tasks list
    from modules.async_worker import async_tasks
    async_tasks.append(task)

    # Wait for the task to complete
    while not task.processing:
        time.sleep(0.1)
    while task.processing:
        time.sleep(0.1)

    return task.results

# Add this to your Gradio interface
import gradio as gr

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