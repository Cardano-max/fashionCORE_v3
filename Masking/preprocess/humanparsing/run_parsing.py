import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch

parsingatr_ckpts_path = os.path.join(PROJECT_ROOT, 'ckpt/humanparsing/')
parsinglip_ckpts_path = os.path.join(PROJECT_ROOT, 'ckpt/humanparsing/')

class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.base_model_path = Path(__file__).absolute().parents[2].absolute() / 'ckpt/humanparsing'
        self.body_model_path = self.base_model_path / 'parsing_atr.onnx'
        self.lip_model_path = self.base_model_path / 'parsing_lip.onnx'
        
        # Define Hugging Face URLs for the models
        self.body_model_url = 'https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx'
        self.lip_model_url = 'https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx'
        
        # Ensure models are downloaded
        self.download_model(self.body_model_path, self.body_model_url)
        self.download_model(self.lip_model_path, self.lip_model_url)

        # Load models into inference sessions
        self.session = ort.InferenceSession(str(self.body_model_path), sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(str(self.lip_model_path), sess_options=session_options, providers=['CPUExecutionProvider'])

    def download_model(self, model_path, model_url):
        """Download the model if it does not exist."""
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            print(f"Model not found at {model_path}. Downloading from {model_url}...")
            load_file_from_url(model_url, model_dir=str(model_path.parent))
            print(f"Model downloaded and saved to {model_path}.")
        else:
            print(f"Model already exists at {model_path}.")


    def __call__(self, input_image):
        # torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
