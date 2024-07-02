# modules/virtual_tryon.py

import gradio as gr
import modules.config
import modules.flags as flags
import modules.html
from modules.rembg import rembg_run
from modules.util import HWC3
from extras.inpaint_mask import generate_mask_from_image
import modules.core as core

def virtual_tryon(person_image, clothes_image):
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
    inputs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'style_selections': ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
        'performance_selection': flags.Performance.QUALITY,
        'aspect_ratios_selection': "1152×896",
        'image_number': 1,
        'image_seed': 0,
        'sharpness': 2.0,
        'guidance_scale': 7.0,
        'base_model_name': modules.config.default_base_model_name,
        'refiner_model_name': modules.config.default_refiner_model_name,
        'refiner_switch': 0.8,
        'sampler_name': modules.config.default_sampler,
        'scheduler_name': modules.config.default_scheduler,
        'overwrite_step': -1,
        'overwrite_switch': -1,
        'mixing_image_prompt_and_inpaint': True,
        'mixing_image_prompt_and_vary_upscale': False,
        'inpaint_engine': 'v2.6',
        'inpaint_strength': 1.0,
        'inpaint_respective_field': 1.0,
    }
    
    # Add Image Prompt and Inpaint inputs
    inputs['uov_input_image'] = ip_image
    inputs['uov_method'] = flags.disabled
    inputs['outpaint_selections'] = []
    inputs['inpaint_input_image'] = {'image': person_image, 'mask': inpaint_mask}
    inputs['inpaint_additional_prompt'] = ''
    
    # Step 6: Call the generation function
    task = core.generate_images(**inputs)
    
    while len(task.yields) == 0:
        task.process_yields()
    
    final_results = []
    for y in task.yields:
        if y[0] == 'results':
            final_results = y[1]
    
    return final_results[0] if final_results else None

def create_virtual_tryon_interface():
    with gr.Blocks() as virtual_tryon_interface:
        with gr.Row():
            person_input = gr.Image(label="Upload Person Image", type="numpy")
            clothes_input = gr.Image(label="Upload Clothes Image", type="numpy")
        generate_btn = gr.Button("Generate Virtual Try-On")
        output_image = gr.Image(label="Result")

        generate_btn.click(virtual_tryon, inputs=[person_input, clothes_input], outputs=output_image)
    
    return virtual_tryon_interface