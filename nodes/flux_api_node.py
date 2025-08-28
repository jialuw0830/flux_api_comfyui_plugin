"""
ComfyUI Node for Eigen AI FLUX API Integration
This node allows ComfyUI to use FLUX.1-schnell model for content generation with LoRA support

Key Features:
- Support for up to 3 LoRAs simultaneously
- Default LoRA configuration, ready to use
- User-friendly interface
- Flexible LoRA weight adjustment
- Real-time status monitoring
- Content upscaling functionality
"""

import json
import base64
import requests
import numpy as np
from PIL import Image
import io
import os
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxAPINode:
    """
    ComfyUI Eigen AI FLUX API Integration Node
    
    This node sends requests to the Eigen AI FLUX API and returns generated images
    that can be used in ComfyUI workflows.
    
    Usage:
    1. Set prompt and image dimensions
    2. Select up to 3 LoRAs and adjust weights
    3. Optionally enable image upscaling
    4. Run the node to generate images
    """
    
    def __init__(self):
        self.api_base_url = "http://74.81.65.108:8000"
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minutes timeout for generation
        logger.info("Websocket connected")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A beautiful landscape painting in Studio Ghibli style",
                    "description": "Text prompt for generation (or connect from FluxPromptNode)",
                    "multiline": True,
                    "max_length": 2000,
                    "display": "textarea"  # Use textarea display for larger input area
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number"
                }),
                "upscale": ("BOOLEAN", {
                    "default": False,
                    "description": "Whether to upscale the generated image"
                }),
                "upscale_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "step": 2,
                    "display": "dropdown"
                }),
                "api_url": ("STRING", {
                    "default": "http://74.81.65.108:8000",
                    "description": "Eigen AI FLUX API base URL"
                }),
                 "lora1_name": ("STRING", {
                    "default": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "description": "First LoRA name (default: Studio Ghibli Flux)",
                    "multiline": False
                }),
                "lora1_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
               
                "lora2_name": ("STRING", {
                    "default": "none",
                    "description": "Second LoRA name (optional, use 'none' to disable)",
                    "multiline": False
                }),
                "lora2_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "lora3_name": ("STRING", {
                    "default": "none",
                    "description": "Third LoRA name (optional, use 'none' to disable)",
                    "multiline": False
                }),
                "lora3_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Eigen AI FLUX API"
    OUTPUT_NODE = False
    
    def generate_image(self, prompt, width, height, seed, upscale, upscale_factor, api_url,
                      lora1_name="/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors", lora1_weight=1.0, 
                      lora2_name="none", lora2_weight=1.0, 
                      lora3_name="none", lora3_weight=1.0):
        """
        Generate content using Eigen AI FLUX API
        
        Args:
            prompt (str): Text prompt for generation
            width (int): Width
            height (int): Height
            seed (int): Random seed (-1 for random)
            upscale (bool): Whether to enable upscaling
            upscale_factor (int): Upscaling factor
            api_url (str): Eigen AI FLUX API base URL
            lora1_name (str): First LoRA name (optional, has default)
            lora1_weight (float): First LoRA weight (optional, has default)
            lora2_name (str): Second LoRA name (optional)
            lora2_weight (float): Second LoRA weight (optional)
            lora3_name (str): Third LoRA name (optional)
            lora3_weight (float): Third LoRA weight (optional)
            
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
                "upscale": upscale,
                "upscale_factor": upscale_factor
            }
            
            # Add seed if specified
            if seed != -1:
                payload["seed"] = seed
            
            # Add LoRA configuration
            loras_to_apply = []
            
            # Add LoRA 1 (required) - always add
            loras_to_apply.append({
                "name": lora1_name.strip(),
                "weight": lora1_weight
            })
            logger.info(f"Added LoRA 1 (required): {lora1_name.strip()} (weight: {lora1_weight})")
            
            # Add LoRA 2 if specified and valid
            if (lora2_name and lora2_name.strip() and 
                lora2_name.strip().lower() != "none" and 
                lora2_name.strip() != ""):
                loras_to_apply.append({
                    "name": lora2_name.strip(),
                    "weight": lora2_weight
                })
                logger.info(f"Added LoRA 2: {lora2_name.strip()} (weight: {lora2_weight})")
            
            # Add LoRA 3 if specified and valid
            if (lora3_name and lora3_name.strip() and 
                lora3_name.strip().lower() != "none" and 
                lora3_name.strip() != ""):
                loras_to_apply.append({
                    "name": lora3_name.strip(),
                    "weight": lora3_weight
                })
                logger.info(f"Added LoRA 3: {lora3_name.strip()} (weight: {lora3_weight})")
            
            # Limit to maximum 3 LoRAs
            if len(loras_to_apply) > 3:
                logger.warning(f"Too many LoRAs specified ({len(loras_to_apply)}), limiting to first 3")
                loras_to_apply = loras_to_apply[:3]
            
            # Add LoRAs to payload (LoRA 1 is always required)
            payload["loras"] = loras_to_apply
            logger.info(f"Applying {len(loras_to_apply)} LoRAs: {loras_to_apply}")
            
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
            
            # Get download URL from response (use download_url instead of image_url)
            download_url = result.get("download_url", "")
            if not download_url:
                raise Exception("No download URL in API response")
            
            # Construct full download URL
            if download_url.startswith("/"):
                full_download_url = f"{self.api_base_url}{download_url}"
            else:
                full_download_url = f"{self.api_base_url}/{download_url}"
            
            logger.info(f"Downloading image from: {full_download_url}")
            
            # Download the generated image using the download endpoint
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
            
            # Additional ComfyUI compatibility checks
            logger.info(f"Tensor device: {image_tensor.device}")
            logger.info(f"Tensor requires grad: {image_tensor.requires_grad}")
            
            # Final tensor validation for ComfyUI
            logger.info(f"Final tensor info:")
            logger.info(f"  - Tensor: {image_tensor}")
            logger.info(f"  - Shape: {image_tensor.shape}")
            logger.info(f"  - Dtype: {image_tensor.dtype}")
            logger.info(f"  - Device: {image_tensor.device}")
            logger.info(f"  - Contiguous: {image_tensor.is_contiguous()}")
            logger.info(f"  - Value range: [{image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}]")
            logger.info(f"  - Memory layout: {image_tensor.stride()}")
            
            # Suppress generation-info output per user request
            logger.info("Image generated successfully (generation-info disabled)")
            
            # ComfyUI expects IMAGE type to be PyTorch tensor
            # Save Image and Preview Image nodes will call .cpu().numpy() internally
            logger.info(f"Returning PyTorch tensor for ComfyUI: {image_tensor.shape}, {image_tensor.dtype}")
            
            return (image_tensor,)
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            
            # Return a placeholder image and error info
            # ComfyUI expects PyTorch tensor
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
        Force re-execution when key parameters change
        """
        # Always regenerate when prompt, dimensions, or LoRA settings change
        lora_hash = f"{kwargs.get('lora1_name', '/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors')}_{kwargs.get('lora1_weight', 1.0)}_{kwargs.get('lora2_name', 'none')}_{kwargs.get('lora2_weight', 1.0)}_{kwargs.get('lora3_name', 'none')}_{kwargs.get('lora3_weight', 1.0)}"
        return f"{kwargs.get('prompt', '')}_{kwargs.get('width', 512)}_{kwargs.get('height', 512)}_{lora_hash}"





# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxAPINode": FluxAPINode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAPINode": "Eigen AI FLUX Schnell API Generator"
}
