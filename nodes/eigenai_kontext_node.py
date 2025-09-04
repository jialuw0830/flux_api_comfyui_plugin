"""
ComfyUI Node for Eigen AI FLUX Kontext API Integration
This node allows ComfyUI to use FLUX.1-Kontext-dev model for image-to-image generation

Key Features:
- Image-to-image generation using FLUX.1-Kontext-dev model
- Support for up to 3 LoRAs simultaneously
- Default LoRA configuration, ready to use
- User-friendly interface
- Flexible LoRA weight adjustment
- Real-time status monitoring
- Content upscaling functionality
- Background removal functionality with adjustable strength
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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EigenAIKontextNode:
    """
    ComfyUI Eigen AI FLUX Kontext API Integration Node
    
    This node sends requests to the Eigen AI FLUX Kontext API and returns generated images
    that can be used in ComfyUI workflows. It performs image-to-image generation.
    
    Usage:
    1. Connect an input image
    2. Set prompt and image dimensions
    3. Select up to 3 LoRAs and adjust weights
    4. Optionally enable image upscaling
    5. Run the node to generate images
    """
    
    def __init__(self):
        self.api_base_url = "http://74.81.65.108:9000"
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minutes timeout for generation
        logger.info("Kontext API Node initialized")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "image": ("IMAGE", {
                    "description": "Input image for image-to-image generation"
                }),
                "prompt": ("STRING", {
                    "description": "Text prompt from text node (connect from EigenAITextNode)"
                }),
                "auto_downscale_large_images": ("BOOLEAN", {
                    "default": True,
                    "description": "Auto-downscale large images (>512px) by half before processing"
                }),

                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number"
                }),
                "inference_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider",
                    "description": "Number of inference steps for generation"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider",
                    "description": "Guidance scale for text conditioning (negative values for negative guidance)"
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
                "enable_background_removal": ("BOOLEAN", {
                    "default": False,
                    "description": "Whether to enable background removal"
                }),
                "removal_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Background removal strength (0.0 = gentle, 1.0 = aggressive)"
                }),
                "api_url": ("STRING", {
                    "default": "http://74.81.65.108:9000",
                    "description": "FLUX Kontext API base URL"
                }),
                "lora1_name": ("STRING", {
                    "default": "21j3h123/realEarthKontext",
                    "description": "First LoRA name (default: realEarthKontext emoji style)",
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
    
    def generate_image(self, image, prompt, auto_downscale_large_images, seed, inference_steps, guidance_scale, upscale, upscale_factor, enable_background_removal, removal_strength, api_url,
                      lora1_name="21j3h123/realEarthKontext", lora1_weight=1.0, 
                      lora2_name="none", lora2_weight=1.0, 
                      lora3_name="none", lora3_weight=1.0):
        """
        Generate content using Eigen AI FLUX Kontext API (image-to-image)
        
        Args:
            image: Input image tensor from ComfyUI
            prompt (str): Text prompt for generation
            auto_downscale_large_images (bool): Auto downscale if either dimension > 512
            seed (int): Random seed (-1 for random)
            inference_steps (int): Number of inference steps for generation
            guidance_scale (float): Guidance scale for text conditioning (negative values for negative guidance)
            upscale (bool): Whether to enable upscaling
            upscale_factor (int): Upscaling factor
            enable_background_removal (bool): Whether to enable background removal
            removal_strength (float): Background removal strength (0.0 = gentle, 1.0 = aggressive)
            api_url (str): FLUX Kontext API base URL
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
            if api_url and api_url != "http://74.81.65.108:9000":
                self.api_base_url = api_url
            
            # Convert ComfyUI image tensor to PIL Image
            if isinstance(image, torch.Tensor):
                # Remove batch dimension if present
                if len(image.shape) == 4:
                    image = image.squeeze(0)
                
                # Convert to numpy and then to PIL
                image_np = image.cpu().numpy()
                # Ensure values are in [0, 255] range
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(image_np)
                logger.info(f"Converted ComfyUI tensor to PIL image: {pil_image.size}, {pil_image.mode}")
            else:
                pil_image = image
            
            # Ensure image is in RGB mode
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                logger.info(f"Converted image to RGB mode")
            
            # Auto-downscale large images if enabled
            try:
                if auto_downscale_large_images:
                    ow, oh = pil_image.size
                    if ow > 512 or oh > 512:
                        nw, nh = max(1, ow // 2), max(1, oh // 2)
                        logger.info(f"Auto-downscaling from {(ow, oh)} to {(nw, nh)}")
                        pil_image = pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
            except Exception as resize_e:
                logger.warning(f"Auto-downscale skipped due to error: {resize_e}")
            
            # Convert PIL image to bytes for API request
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Prepare form data for the API request
            files = {
                'image': ('input_image.png', img_byte_arr, 'image/png')
            }
            
            # Use processed image size for width/height
            pw, ph = pil_image.size
            data = {
                'prompt': prompt,
                'width': pw,
                'height': ph,
                'upscale': upscale,
                'upscale_factor': upscale_factor,
                'enable_background_removal': enable_background_removal,
                'removal_strength': removal_strength
            }
            
            # Add seed if specified
            if seed != -1:
                data['seed'] = seed
            
            # Add inference steps and guidance scale
            data['inference_steps'] = inference_steps
            data['guidance_scale'] = guidance_scale
            
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
            
            # Add LoRAs to data (LoRA 1 is always required)
            data["loras"] = json.dumps(loras_to_apply)
            logger.info(f"Applying {len(loras_to_apply)} LoRAs: {loras_to_apply}")
            
            logger.info(f"Sending request to FLUX Kontext API: {self.api_base_url}/generate-with-image-and-return")
            logger.info(f"Data: {data}")
            
            # Make API request to the image-to-image endpoint
            response = self.session.post(
                f"{self.api_base_url}/generate-with-image-and-return",
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Check if response is image data (direct image return)
            content_type = response.headers.get('content-type', '')
            logger.info(f"Response content type: {content_type}")
            
            if 'image/' in content_type:
                # Direct image response - use response content directly
                logger.info(f"Received direct image response, size: {len(response.content)} bytes")
                image_data = response.content
            else:
                # JSON response with download URL (fallback)
                logger.info("Received JSON response, attempting to parse")
                try:
                    result = response.json()
                    logger.info(f"API response received: {result.get('message', 'Success')}")
                    logger.info(f"Full API response result: {result}")
                    
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
                    
                    image_data = image_response.content
                    logger.info(f"Downloaded image data size: {len(image_data)} bytes")
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON decode error: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    raise Exception(f"Invalid response from API: {json_error}")
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
            
            # Suppress generation-info output per user request
            logger.info("Kontext image generated successfully (generation-info disabled)")
            
            # ComfyUI expects IMAGE type to be PyTorch tensor
            logger.info(f"Returning PyTorch tensor for ComfyUI: {image_tensor.shape}, {image_tensor.dtype}")
            
            return (image_tensor,)
            
        except Exception as e:
            error_msg = f"Error generating Kontext image: {str(e)}"
            logger.error(error_msg)
            
            # Infer placeholder size from input image if possible
            ph, pw = 512, 512
            try:
                if isinstance(image, torch.Tensor) and len(image.shape) == 4:
                    ph, pw = int(image.shape[1]), int(image.shape[2])
                elif isinstance(image, Image.Image):
                    pw, ph = image.size
            except Exception:
                pass
            placeholder = np.zeros((ph, pw, 3), dtype=np.float32)
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
        # Always regenerate when prompt or LoRA settings change (no explicit dimensions)
        lora_hash = f"{kwargs.get('lora1_name', '21j3h123/realEarthKontext')}_{kwargs.get('lora1_weight', 1.0)}_{kwargs.get('lora2_name', 'none')}_{kwargs.get('lora2_weight', 1.0)}_{kwargs.get('lora3_name', 'none')}_{kwargs.get('lora3_weight', 1.0)}"
        return f"{kwargs.get('prompt', '')}_{kwargs.get('seed', -1)}_{kwargs.get('guidance_scale', 7.5)}_{kwargs.get('auto_downscale_large_images', True)}_{lora_hash}"





# Node class mappings
NODE_CLASS_MAPPINGS = {
    "EigenAIKontextNode": EigenAIKontextNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "EigenAIKontextNode": "Eigen AI FLUX Kontext API Generator"
}
