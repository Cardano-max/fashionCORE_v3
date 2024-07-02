# modules/virtual_tryon.py

import gradio as gr
import modules.config
import modules.flags as flags
import modules.html
from modules.rembg import rembg_run
from modules.util import HWC3
from extras.inpaint_mask import generate_mask_from_image
import modules.async_worker as worker
import time
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def virtual_tryon(person_image, clothes_image):
    try:
        logger.info("Starting virtual try-on process")
        
        if person_image is None or clothes_image is None:
            raise ValueError("Both person and clothes images are required")
        
        # Step 1: Remove background from clothes image
        clothes_no_bg = rembg_run(clothes_image)
        
        # Step 2: Set up Image Prompt
        ip_image = HWC3(clothes_no_bg)
        ip_stop = 0.6  # Default value
        ip_weight = 0.6  # Default value
        
        # Step 3: Generate mask for person image
        mask_extras = {
            'sam_model': 'sam_vit_b_01ec64',
            'sam_prompt_text': 'Clothes',
            'box_threshold': 0.3,
            'text_threshold': 0.25,
            'sam_quant': False
        }
        inpaint_mask = generate_mask_from_image(person_image, 'sam', mask_extras)
        
        # Step 4: Set up parameters
        prompt = "A person wearing the clothes from the reference image"
        negative_prompt = "Unrealistic, blurry, low quality"
        
        # Step 5: Prepare inputs for the generation function
        inputs = [
            prompt,
            negative_prompt,
            False,  # translate_prompts
            ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],  # style_selections
            flags.Performance.QUALITY.value,
            "1152×896",  # aspect_ratios_selection
            1,  # image_number
            'png',  # output_format
            0,  # image_seed
            2.0,  # sharpness
            7.0,  # guidance_scale
            modules.config.default_base_model_name,
            modules.config.default_refiner_model_name,
            0.8,  # refiner_switch
            modules.config.default_loras,
            True,  # input_image_checkbox
            'ip',  # current_tab
            flags.disabled,  # uov_method
            ip_image,
            [],  # outpaint_selections
            {'image': person_image, 'mask': inpaint_mask},  # inpaint_input_image
            '',  # inpaint_additional_prompt
            None,  # inpaint_mask_image_upload
            False, False, False,  # disable_preview, disable_intermediate_results, black_out_nsfw
            1.5, 0.8, 0.3,  # adm_scaler_positive, adm_scaler_negative, adm_scaler_end
            7.0,  # adaptive_cfg
            modules.config.default_sampler,
            modules.config.default_scheduler,
            -1, -1, -1, -1, -1,  # overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength
            -1,  # overwrite_upscale_strength
            False, True,  # mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint
            False, False,  # debugging_cn_preprocessor, skipping_cn_preprocessor
            64, 128,  # canny_low_threshold, canny_high_threshold
            flags.refiner_swap_method,
            0.25,  # controlnet_softness
            False, 1.01, 1.02, 0.99, 0.95,  # freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2
        ]
        
        # Step 6: Create and process the task
        task = worker.AsyncTask(args=inputs)
        worker.async_tasks.append(task)
        
        # Wait for the task to complete
        while len(task.yields) == 0:
            time.sleep(0.01)
        
        # Retrieve the results
        final_results = []
        for y in task.yields:
            if y[0] == 'results':
                final_results = y[1]
        
        logger.info("Virtual try-on process completed successfully")
        return final_results[0] if final_results else None
    except Exception as e:
        logger.error(f"Error in virtual try-on process: {str(e)}")
        logger.debug(traceback.format_exc())
        return gr.update(value=None), gr.update(value=f"Error: {str(e)}")

def create_virtual_tryon_interface():
    with gr.Blocks() as virtual_tryon_interface:
        with gr.Row():
            person_input = gr.Image(label="Upload Person Image", type="numpy")
            clothes_input = gr.Image(label="Upload Clothes Image", type="numpy")
        generate_btn = gr.Button("Generate Virtual Try-On")
        output_image = gr.Image(label="Result")
        error_output = gr.Textbox(label="Error", visible=False)

        generate_btn.click(virtual_tryon, 
                           inputs=[person_input, clothes_input], 
                           outputs=[output_image, error_output])
    
    return virtual_tryon_interface