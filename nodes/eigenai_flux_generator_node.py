"""
ComfyUI Node for Eigen AI FLUX Image Generation
This node handles image generation using FLUX API with inputs from text and LoRA nodes

Key Features:
- Image generation with FLUX API
- Receives processed text from text node
- Receives LoRA config from LoRA node
- Handles generation parameters (dimensions, seed, guidance)
- Image upscaling support
"""

import json
import requests
import numpy as np
from PIL import Image
import io
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EigenAIFluxGeneratorNode:
    """
    ComfyUI Eigen AI FLUX Image Generation Node
    
    This node generates images using the FLUX API. It receives processed text from the text node
    and LoRA configuration from the LoRA node, focusing solely on image generation.
    
    Usage:
    1. Connect text inputs from EigenAIFluxTextNode
    2. Connect LoRA config from EigenAIFluxLoraNode
    3. Set generation parameters (dimensions, seed, guidance)
    4. Run to generate images
    """
    
    def __init__(self):
        self.api_base_url = "http://74.81.65.108:8000"
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minutes timeout for generation
        logger.info("EigenAI FLUX Generator Node initialized")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "prompt": ("PROMPT", {
                    "description": "Text prompt from text node (connect from EigenAIFluxTextNode)",
                    "default": "",
                    "multiline": False
                }),
                "lora_config": ("LORA_CONFIG", {
                    "description": "LoRA configuration from LoRA node"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider",
                    "description": "Image width"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider",
                    "description": "Image height"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number",
                    "description": "Random seed (-1 for random)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Guidance scale for generation"
                }),

                "api_url": ("STRING", {
                    "default": "http://74.81.65.108:8000",
                    "description": "Eigen AI FLUX API base URL"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Eigen AI Modular"
    OUTPUT_NODE = False
    
    def generate_image(self, prompt, lora_config, width, height, seed, 
                      guidance_scale, api_url):
        """
        Generate image using Eigen AI FLUX API
        
        Args:
            prompt (str): Text prompt from text node
            lora_config (dict): LoRA configuration from LoRA node
            width (int): Image width
            height (int): Image height
            seed (int): Random seed (-1 for random)
            guidance_scale (float): Guidance scale
            api_url (str): API base URL
            
        Returns:
            image_tensor (IMAGE)
        """
        try:
            # Update API URL if provided
            if api_url and api_url != "http://74.81.65.108:8000":
                self.api_base_url = api_url
            
            # Prepare request payload
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale
            }
            
            # Add seed if specified
            if seed != -1:
                payload["seed"] = seed
            
            # Add LoRA configuration from LoRA node
            if lora_config and "loras" in lora_config:
                payload["loras"] = lora_config["loras"]
                logger.info(f"Using LoRA config: {lora_config['total_count']} LoRAs")
            else:
                logger.warning("No LoRA config provided, using default")
                payload["loras"] = [{
                    "name": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "weight": 1.0
                }]
            
            logger.info(f"Sending request to Eigen AI FLUX API: {self.api_base_url}/generate")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Make API request
            response = self.session.post(
                f"{self.api_base_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse response
            result = response.json()
            logger.info(f"API response received: {result.get('message', 'Success')}")
            
            # Get download URL from response
            download_url = result.get("download_url", "")
            if not download_url:
                raise Exception("No download URL in API response")
            
            # Construct full download URL
            if download_url.startswith("/"):
                full_download_url = f"{self.api_base_url}{download_url}"
            else:
                full_download_url = f"{self.api_base_url}/{download_url}"
            
            logger.info(f"Downloading image from: {full_download_url}")
            
            # Download the generated image
            try:
                image_response = self.session.get(full_download_url, timeout=60)
                logger.info(f"Image download response status: {image_response.status_code}")
            except Exception as download_error:
                logger.error(f"Image download failed: {download_error}")
                raise Exception(f"Failed to download image: {download_error}")
            
            if image_response.status_code != 200:
                raise Exception(f"Failed to download image: {image_response.status_code}")
            
            # Convert image to PIL Image
            image_data = image_response.content
            logger.info(f"Downloaded image data size: {len(image_data)} bytes")
            
            pil_image = Image.open(io.BytesIO(image_data))
            logger.info(f"PIL Image mode: {pil_image.mode}, size: {pil_image.size}")
            
            # Convert to RGB if necessary
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                logger.info(f"Converted to RGB mode")
            
            # Convert to numpy array and then to PyTorch tensor (ComfyUI format)
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            logger.info(f"Numpy array shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            # Add batch dimension if needed
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, 0)
                logger.info(f"Added batch dimension, new shape: {image_array.shape}")
            
            # Convert to PyTorch tensor (ComfyUI expects this format)
            image_tensor = torch.from_numpy(image_array)
            logger.info(f"PyTorch tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            
            # Ensure tensor is contiguous and in the right format for ComfyUI
            image_tensor = image_tensor.contiguous()
            logger.info(f"Tensor is contiguous: {image_tensor.is_contiguous()}")
            
            # Verify tensor values
            logger.info(f"Tensor min value: {image_tensor.min().item()}, max value: {image_tensor.max().item()}")
            
            logger.info("Image generated successfully")
            
            return (image_tensor,)
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            
            # Return a placeholder image and error info
            placeholder = np.zeros((height, width, 3), dtype=np.float32)
            placeholder[:, :, 0] = 0.8  # Light red for error
            # Add batch dimension for ComfyUI compatibility
            placeholder = np.expand_dims(placeholder, 0)
            
            # Convert to PyTorch tensor for ComfyUI compatibility
            placeholder_tensor = torch.from_numpy(placeholder)
            logger.info(f"Returning error placeholder tensor: {placeholder_tensor.shape}, {placeholder_tensor.dtype}")
            return (placeholder_tensor,)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        """
        Force re-execution when generation parameters change
        """
        return f"{kwargs.get('prompt', '')}_{kwargs.get('width', 512)}_{kwargs.get('height', 512)}_{kwargs.get('seed', -1)}_{kwargs.get('guidance_scale', 3.5)}"
