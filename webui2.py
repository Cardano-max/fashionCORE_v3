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
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from modules.flags import Performance
from queue import Queue
from threading import Lock, Event, Thread
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_cor = image_to_base64("images/corre.jpg")
base64_inc = image_to_base64("images/inc.jpg")

# Set up environment variables for sharing data
os.environ['GRADIO_PUBLIC_URL'] = ''
os.environ['GENERATED_IMAGE_PATH'] = ''
os.environ['MASKED_IMAGE_PATH'] = ''

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = custom_exception_handler

# Initialize Segformer model and processor
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

# Initialize queue and locks
task_queue = Queue()
queue_lock = Lock()
current_task_event = Event()
queue_update_event = Event()

# Function to send email (using Mailpit for demonstration)
def send_feedback_email(rating, comment):
    sender_email = "feedback@arbitryon.com"
    receiver_email = "feedback@arbitryon.com"  # This would be your actual feedback collection email
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"ArbiTryOn Feedback - Rating: {rating}"

    body = f"Rating: {rating}/5\nComment: {comment}"
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("localhost", 1025) as server:  # Mailpit default settings
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Feedback email sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send feedback email: {str(e)}")
        return False


def generate_mask(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    labels = [4, 14, 15, 6, 12, 13]  # Upper Clothes 4, Left Arm 14, Right Arm 15

    combined_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in labels:
        combined_mask = torch.logical_or(combined_mask, pred_seg == label)

    pred_seg_new = torch.zeros_like(pred_seg)
    pred_seg_new[combined_mask] = 255

    image_mask = pred_seg_new.numpy().astype(np.uint8)

    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(image_mask, kernel, iterations=1)

    return dilated_mask

def virtual_try_on(clothes_image, person_image):
    try:
        # Convert person_image to PIL Image
        person_pil = Image.fromarray(person_image)

        # Generate mask
        inpaint_mask = generate_mask(person_pil)

        # Resize images and mask
        target_size = (1024, 1024)
        clothes_image = HWC3(clothes_image)
        person_image = HWC3(person_image)
        inpaint_mask = HWC3(inpaint_mask)[:, :, 0]

        clothes_image = resize_image(clothes_image, target_size[0], target_size[1])
        person_image = resize_image(person_image, target_size[0], target_size[1])
        inpaint_mask = resize_image(inpaint_mask, target_size[0], target_size[1])

        # Display and save the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(inpaint_mask, cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        masked_image_path = os.path.join(modules.config.path_outputs, f"masked_image_{int(time.time())}.png")
        with open(masked_image_path, 'wb') as f:
            f.write(buf.getvalue())
        
        plt.close()

        os.environ['MASKED_IMAGE_PATH'] = masked_image_path

        loras = []
        for lora in modules.config.default_loras:
            loras.extend(lora)

        args = [
            True,
            "",
            modules.config.default_prompt_negative,
            False,
            modules.config.default_styles,
            Performance.QUALITY.value,
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
            False,
            False,
            0,
            modules.config.default_save_metadata_to_images,
            modules.config.default_metadata_scheme,
        ]

        args.extend([
            clothes_image,
            0.86,
            0.97,
            flags.default_ip,
        ])

        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)

        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            return {"success": True, "image_path": task.results[0], "masked_image_path": masked_image_path}
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        print("Error in virtual_try_on:", str(e))
        traceback.print_exc()
        return {"success": False, "error": str(e)}

example_garments = [
    "images/b1.png",
    "images/b2.jpeg",
    "images/b3.jpeg",
    "images/b4.jpeg",
    "images/b5.jpeg",
    "images/b6.jpeg",
    "images/b7.png",
    "images/b8.png",
    "images/b9.png",
    "images/b10.png",
    "images/b11.png",
    "images/b12.png",
    "images/b13.png",
    "images/b14.jpg",
    "images/b15.png",
    "images/b16.png",
    "images/b17.png",
    "images/b18.png",

]

# Subtle loading messages
loading_messages = [
    "Preparing your virtual fitting room...",
    "Analyzing fashion possibilities...",
    "Calculating style quotient...",
    "Assembling your digital wardrobe...",
    "Initiating virtual try-on sequence...",
]

# Gentle error messages
error_messages = [
    "Oops! We've hit a small snag. Our team is on it!",
    "It seems our digital tailor needs a quick coffee break. We'll be right back!",
    "Minor hiccup in the fashion matrix. We're working to smooth it out.",
    "Looks like we're experiencing a slight style delay. Thanks for your patience!",
    "Our virtual dressing room is temporarily out of order. We're fixing it up!",
]


css = """
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;700&display=swap');

    body, .gradio-container {
        background-color: #000810;
        color: #e0fbfc;
        font-family: 'Roboto', sans-serif;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(6, 190, 225, 0.05) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255, 83, 168, 0.05) 0%, transparent 20%);
        background-attachment: fixed;
    }

    .header {
        background: linear-gradient(135deg, #0a192f 0%, #0e2b50 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3), 
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
    }

    .header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0.1) 0%,
            rgba(255, 255, 255, 0.05) 40%,
            rgba(255, 255, 255, 0) 50%
        );
        transform: rotate(-45deg);
        pointer-events: none;
    }

    .title {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 10px rgba(6, 190, 225, 0.5);
    }

    .subtitle {
        font-size: 22px;
        color: #98c1d9;
        font-weight: 300;
        letter-spacing: 1px;
    }

    .example-garments img {
        border-radius: 15px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border: 2px solid transparent;
    }

    .example-garments img:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 20px 40px rgba(6, 190, 225, 0.3);
        border-color: #06bee1;
    }

    .try-on-button {
        background: linear-gradient(135deg, #06bee1 0%, #0030ff 100%);
        color: white;
        padding: 18px 36px;
        border: none;
        border-radius: 50px;
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }

    .try-on-button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0.3) 0%,
            rgba(255, 255, 255, 0.1) 40%,
            rgba(255, 255, 255, 0) 50%
        );
        transform: rotate(-45deg);
        transition: all 0.3s ease;
    }

    .try-on-button:hover {
        box-shadow: 0 10px 20px rgba(6, 190, 225, 0.4);
    }

    .try-on-button:hover::before {
        left: 100%;
    }

    .queue-info, .info-message {
        background-color: rgba(6, 190, 225, 0.1);
        border: 1px solid #06bee1;
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        font-size: 18px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }

    .error-message {
        background-color: rgba(255, 83, 168, 0.1);
        border: 1px solid #ff53a8;
        color: #ff53a8;
    }

    .result-links a {
        color: #06bee1;
        text-decoration: none;
        margin: 0 20px;
        transition: all 0.3s ease;
        font-weight: 500;
        position: relative;
    }

    .result-links a::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 2px;
        bottom: -5px;
        left: 0;
        background-color: #06bee1;
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .result-links a:hover {
        color: #ff53a8;
    }

    .result-links a:hover::after {
        transform: scaleX(1);
        background-color: #ff53a8;
    }

    .loading {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(6, 190, 225, 0.3);
        border-radius: 50%;
        border-top-color: #06bee1;
        animation: spin 1s cubic-bezier(0.55, 0.055, 0.675, 0.19) infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .instruction-images {
        display: flex;
        justify-content: space-around;
        margin-top: 40px;
        background: rgba(255, 255, 255, 0.03);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }

    .instruction-image {
        text-align: center;
        max-width: 45%;
    }

    .instruction-image img {
        max-width: 100%;
        border-radius: 15px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.4s ease;
    }

    .instruction-image img:hover {
        transform: scale(1.05) rotate(2deg);
        box-shadow: 0 20px 40px rgba(6, 190, 225, 0.4);
    }

    .instruction-caption {
        margin-top: 20px;
        font-size: 18px;
        color: #98c1d9;
        font-weight: 400;
        letter-spacing: 1px;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }

    ::-webkit-scrollbar-track {
        background: #0a192f;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #06bee1 0%, #0030ff 100%);
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0030ff 0%, #06bee1 100%);
    }

    /* Gradio-specific customizations */
    .gr-button-primary {
        background: linear-gradient(135deg, #06bee1 0%, #0030ff 100%) !important;
        border: none !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    .gr-input, .gr-dropdown {
        background-color: #0a192f !important;
        border: 1px solid #06bee1 !important;
        color: #e0fbfc !important;
        border-radius: 10px !important;
    }

    .gr-form {
        background-color: rgba(10, 25, 47, 0.7) !important;
        border: 1px solid rgba(6, 190, 225, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
    }

    .gr-form:focus-within {
        box-shadow: 0 0 20px rgba(6, 190, 225, 0.5);
    }

    /* Animation for loading text */
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    .loading-text {
        animation: pulse 1.5s infinite;
        font-family: 'Orbitron', sans-serif;
        color: #06bee1;
    }
"""
def process_queue():
    while True:
        task = task_queue.get()
        if task is None:
            break
        clothes_image, person_image, result_callback = task
        current_task_event.set()
        queue_update_event.set()
        result = virtual_try_on(clothes_image, person_image)
        current_task_event.clear()
        result_callback(result)
        task_queue.task_done()
        queue_update_event.set()

# Start the queue processing thread
queue_thread = Thread(target=process_queue, daemon=True)
queue_thread.start()

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.HTML(
        """
        <div class="header">
            <h1 class="title">ArbiTryOn: Virtual Fitting Room</h1>
            <p class="subtitle">Experience the future of online shopping with our AI-powered try-on system</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Step 1: Select a Garment")
            example_garment_gallery = gr.Gallery(value=example_garments, columns=2, rows=2, label="Example Garments", elem_class="example-garments")
            clothes_input = gr.Image(label="Selected Garment", source="upload", type="numpy")

        with gr.Column(scale=3):
            gr.Markdown("### Step 2: Upload Your Photo")
            person_input = gr.Image(label="Your Photo", source="upload", type="numpy")

    gr.HTML(f"""
        <div class="instruction-images">
            <div class="instruction-image">
                <img src="data:image/jpg;base64,{base64_cor}" alt="Correct pose">
                <p class="instruction-caption">✅ Correct: Neutral pose, facing forward</p>
            </div>
            <div class="instruction-image">
                <img src="data:image/jpeg;base64,{base64_inc}" alt="Incorrect pose">
                <p class="instruction-caption">❌ Incorrect: Angled or complex pose</p>
            </div>
        </div>
    """)

    try_on_button = gr.Button("Try It On!", elem_classes="try-on-button")
    loading_indicator = gr.HTML(visible=False)
    status_info = gr.HTML(visible=False, elem_classes="info-message")
    masked_output = gr.Image(label="Mask Visualization", visible=False)
    try_on_output = gr.Image(label="Virtual Try-On Result", visible=False)
    image_link = gr.HTML(visible=True, elem_classes="result-links")
    error_output = gr.HTML(visible=False, elem_classes="info-message")

    queue_note = gr.HTML(
        """
        <div class="info-message">
            <p>
                <strong>Note:</strong> If you see a queue status, your request may take extra time to process. 
                We appreciate your patience as we work through all requests.
            </p>
        </div>
        """,
        visible=True
    )

    # Feedback components
    with gr.Row(visible=False) as feedback_row:
        rating = gr.Slider(minimum=1, maximum=5, step=1, label="Rate your experience (1-5 stars)")
        comment = gr.Textbox(label="Leave a comment (optional)")
        submit_feedback = gr.Button("Submit Feedback")

    feedback_status = gr.HTML(visible=False)

    def select_example_garment(evt: gr.SelectData):
        return example_garments[evt.index]

    example_garment_gallery.select(select_example_garment, None, clothes_input)

    def process_virtual_try_on(clothes_image, person_image):
        if clothes_image is None or person_image is None:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                error_output: gr.update(value="<p>Please upload both a garment image and a person image to proceed.</p>", visible=True),
                image_link: gr.update(visible=False),
                queue_note: gr.update(visible=True),
                feedback_row: gr.update(visible=False)
            }
            return

        if current_task_event.is_set():
            yield {
                loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                status_info: gr.update(value="<p>Your request has been queued. We'll process it as soon as possible.</p>", visible=True),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                error_output: gr.update(visible=False),
                image_link: gr.update(visible=False),
                queue_note: gr.update(visible=True)
            }

        def result_callback(result):
            nonlocal generation_done, generation_result
            generation_done = True
            generation_result = result

        with queue_lock:
            current_position = task_queue.qsize()
            task_queue.put((clothes_image, person_image, result_callback))

        generation_done = False
        generation_result = None

        while not generation_done:
            if current_task_event.is_set() and current_position == 0:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value="<p>Processing your request. This may take a few minutes.</p>", visible=True),
                    masked_output: gr.update(visible=False),
                    try_on_output: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                    queue_note: gr.update(visible=True)
                }
            elif current_position > 0:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value=f"<p>Your request is in queue. Current position: {current_position}</p><p>Estimated wait time: {current_position * 2} minutes</p>", visible=True),
                    masked_output: gr.update(visible=False),
                    try_on_output: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                    queue_note: gr.update(visible=True)
                }
            else:
                yield {
                    loading_indicator: gr.update(visible=True, value=f"<div class='loading'></div><p>{random.choice(loading_messages)}</p>"),
                    status_info: gr.update(value="<p>Your request is next in line. Processing will begin shortly.</p>", visible=True),
                    masked_output: gr.update(visible=False),
                    try_on_output: gr.update(visible=False),
                    error_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                    queue_note: gr.update(visible=True)
                }
            
            queue_update_event.wait(timeout=5)
            queue_update_event.clear()
            current_position = max(0, task_queue.qsize() - 1)

        if generation_result is None:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                image_link: gr.update(visible=False),
                error_output: gr.update(value=f"<p>{random.choice(error_messages)}</p><p>Remember, we're still in beta. We appreciate your understanding as we work to improve our service.</p>", visible=True),
                queue_note: gr.update(visible=True)
            }
        elif generation_result['success']:
            generated_image_path = os.environ['GENERATED_IMAGE_PATH']
            masked_image_path = os.environ['MASKED_IMAGE_PATH']
            gradio_url = os.environ['GRADIO_PUBLIC_URL']

            if gradio_url and generated_image_path and masked_image_path:
                output_image_link = f"{gradio_url}/file={generated_image_path}"
                masked_image_link = f"{gradio_url}/file={masked_image_path}"
                link_html = f'<a href="{output_image_link}" target="_blank">View Try-On Result</a> | <a href="{masked_image_link}" target="_blank">View Mask Visualization</a>'

                yield {
                    loading_indicator: gr.update(visible=False),
                    status_info: gr.update(value="<p>Your virtual try-on is complete! Check out the results below.</p>", visible=True),
                    masked_output: gr.update(value=masked_image_path, visible=True),
                    try_on_output: gr.update(value=generated_image_path, visible=True),
                    image_link: gr.update(value=link_html, visible=True),
                    error_output: gr.update(visible=False),
                    queue_note: gr.update(visible=False),
                    feedback_row: gr.update(visible=True)
                }
            else:
                yield {
                    loading_indicator: gr.update(visible=False),
                    status_info: gr.update(visible=False),
                    masked_output: gr.update(visible=False),
                    try_on_output: gr.update(visible=False),
                    image_link: gr.update(visible=False),
                    error_output: gr.update(value="<p>We encountered an issue while generating your try-on results. Our team has been notified and is working on a solution. Please try again later.</p>", visible=True),
                    queue_note: gr.update(visible=True),
                    feedback_row: gr.update(visible=False)
                }
        else:
            yield {
                loading_indicator: gr.update(visible=False),
                status_info: gr.update(visible=False),
                masked_output: gr.update(visible=False),
                try_on_output: gr.update(visible=False),
                image_link: gr.update(visible=False),
                error_output: gr.update(value=f"<p>An error occurred: {generation_result['error']}</p><p>Our team has been notified and is working on a solution. We appreciate your patience as we improve our beta service.</p>", visible=True),
                queue_note: gr.update(visible=True),
                feedback_row: gr.update(visible=False)
            }

    try_on_button.click(
        process_virtual_try_on,
        inputs=[clothes_input, person_input],
        outputs=[loading_indicator, status_info, masked_output, try_on_output, image_link, error_output, queue_note, feedback_row]
    )

    def submit_user_feedback(rating, comment):
        if send_feedback_email(rating, comment):
            return gr.update(value="Thank you for your feedback!", visible=True)
        else:
            return gr.update(value="Failed to submit feedback. Please try again later.", visible=True)

    submit_feedback.click(
        submit_user_feedback,
        inputs=[rating, comment],
        outputs=feedback_status
    )

    gr.Markdown(
        """
        ## How It Works
        1. Choose a garment from our collection or upload your own.
        2. Upload a photo of yourself in a neutral pose (see instructions above).
        3. Click "Try It On!" and let our AI work its magic.

        Please note: ArbiTryOn is currently in beta. While we strive for accuracy, results may vary. 
        We're continuously working to improve the experience and appreciate your feedback!
        """
    )

demo.queue()

def custom_launch():
    app, local_url, share_url = demo.launch(share=True, prevent_thread_lock=True)
    
    if share_url:
        os.environ['GRADIO_PUBLIC_URL'] = share_url
        print(f"Running on public URL: {share_url}")
    
    return app, local_url, share_url

custom_launch()

# Keep the script running
while True:
    time.sleep(1)