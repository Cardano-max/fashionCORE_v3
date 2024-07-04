import gradio as gr
import random
import time
import numpy as np
import modules.config
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
from modules.util import HWC3, resize_image

def virtual_try_on(clothes_image, person_image):
    try:
        clothes_image = HWC3(clothes_image)
        person_image_data = person_image['image'] if isinstance(person_image, dict) else person_image
        person_image = HWC3(person_image_data)
        
        # Assume mask is not needed for now, remove if causing issues
        # inpaint_mask = HWC3(person_image['mask'])[:, :, 0] if isinstance(person_image, dict) and 'mask' in person_image else np.zeros_like(person_image)[:,:,0]

        target_size = (512, 512)
        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])

        args = [
            True,  # generate_image_grid
            "A person wearing new clothes",  # prompt
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
        ]

        # Add LoRA configs
        for lora in modules.config.default_loras:
            args.extend(lora)

        # Add more arguments as needed
        args.extend([
            True,  # input_image_checkbox
            "inpaint",  # current_tab
            flags.disabled,  # uov_method
            None,  # uov_input_image
            [],  # outpaint_selections
            {'image': person_image},  # inpaint_input_image
            "Wearing a new garment",  # inpaint_additional_prompt
            None,  # inpaint_mask_image_upload
            False,  # disable_preview
            False,  # disable_intermediate_results
            modules.config.default_black_out_nsfw,  # black_out_nsfw
            clothes_image,  # ip_adapter_image
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            return task.results[0]
        else:
            return None

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        return None

example_garments = [
    "images/first.png",
    "images/second.png",
    "images/third.png",
    "images/third.png",
]

with gr.Blocks() as demo:
    gr.HTML("<h1>ArbiTryOn - Virtual Try-On System</h1>")

    with gr.Row():
        with gr.Column():
            clothes_input = gr.Image(label="Garment Image", source="upload", type="numpy")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=2, rows=2, label="Example Garments")
        
        with gr.Column():
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy")

    try_on_button = gr.Button("Try It On!")
    try_on_output = gr.Image(label="Virtual Try-On Result")

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    try_on_button.click(
        virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[try_on_output]
    )

demo.launch(share=True)