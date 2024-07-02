import numpy as np
from modules.rembg import rembg_run
from extras.inpaint_mask import generate_mask_from_image
import modules.flags as flags
from modules.async_worker import AsyncTask
import time
import modules.config
import modules.constants as constants
import random
import args_manager
import traceback
import gradio as gr

def remove_background(image):
    try:
        return rembg_run(image)
    except Exception as e:
        print("Error in remove_background:", str(e))
        traceback.print_exc()
        return None

def generate_person_mask(image):
    try:
        result = generate_mask_from_image(
            image,
            'sam',
            {
                'sam_prompt_text': 'Clothes',
                'sam_model': 'sam_vit_b_01ec64',
                'sam_quant': False,
                'box_threshold': 0.3,
                'text_threshold': 0.25
            }
        )
        if result.ndim == 3:
            result = result[:, :, 0]
        elif result.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {result.shape}")
        return result
    except Exception as e:
        print("Error in generate_person_mask:", str(e))
        traceback.print_exc()
        return None

def virtual_tryon(clothes_image, person_image):
    try:
        clothes_no_bg = remove_background(clothes_image)
        if clothes_no_bg is None:
            return "Error in removing background from clothes image."

        person_mask = generate_person_mask(person_image)
        if person_mask is None:
            return "Error in generating mask for person image."

        print(f"Person mask shape: {person_mask.shape}")

        if person_mask.ndim == 3:
            person_mask = person_mask[:, :, 0]
        elif person_mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {person_mask.shape}")

        seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        default_stop, default_weight = flags.default_parameters[flags.cn_ip]

        args = [
            True,
            "A person wearing the clothes from the reference image",
            "Unrealistic, blurry, low quality",
            False,
            ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
            flags.Performance.QUALITY.value,
            modules.config.default_aspect_ratio,
            1,
            modules.config.default_output_format,
            seed,
            modules.config.default_sample_sharpness,
            modules.config.default_cfg_scale,
            modules.config.default_base_model_name,
            modules.config.default_refiner_model_name,
            modules.config.default_refiner_switch,
        ]

        for lora in modules.config.default_loras:
            args.extend(lora)

        args.extend([
            True,
            "inpaint",
            flags.disabled,
            clothes_no_bg,
            [],
            {'image': person_image, 'mask': person_mask},
            "",
            person_mask,
            False,
            False,
            modules.config.default_black_out_nsfw,
            1.0,
            1.0,
            0.0,
            modules.config.default_cfg_tsnr,
            modules.config.default_sampler,
            modules.config.default_scheduler,
            modules.config.default_overwrite_step,
            modules.config.default_overwrite_switch,
            -1,
            -1,
            -1,
            modules.config.default_overwrite_upscale,
            True,
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
        ])

        if not args_manager.args.disable_metadata:
            args.extend([
                modules.config.default_save_metadata_to_images,
                modules.config.default_metadata_scheme,
            ])

        for _ in range(flags.controlnet_image_count):
            args.extend([
                clothes_no_bg,
                default_stop,
                default_weight,
                flags.cn_ip,
            ])

        print(f"Number of arguments: {len(args)}")
        task = AsyncTask(args)

        from modules.async_worker import async_tasks
        async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        return task.results
    except Exception as e:
        print("Error in virtual_tryon:", str(e))
        traceback.print_exc()
        return f"Error: {str(e)}"

def create_virtual_tryon_interface():
    with gr.Blocks() as virtual_tryon_interface:
        with gr.Row():
            clothes_input = gr.Image(label="Clothes Image", type="numpy")
            person_input = gr.Image(label="Person Image", type="numpy")
        generate_btn = gr.Button("Generate Virtual Try-On")
        output_gallery = gr.Gallery(label="Results")

        generate_btn.click(
            virtual_tryon,
            inputs=[clothes_input, person_input],
            outputs=[output_gallery]
        )

    return virtual_tryon_interface