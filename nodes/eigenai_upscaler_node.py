"""
ComfyUI Node for Eigen AI FLUX Image Upscaling
This node handles image upscaling with various methods

Key Features:
- Image upscaling with multiple methods
- Configurable upscale factors
- Quality preservation
- Integration with FLUX workflow
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EigenAIUpscalerNode:
    """
    ComfyUI Eigen AI FLUX Image Upscaler Node
    
    This node handles image upscaling with various methods. It receives images from
    generation nodes and applies upscaling techniques.
    
    Usage:
    1. Connect input image from generation node
    2. Select upscale method and factor
    3. Run to get upscaled image
    """
    
    def __init__(self):
        logger.info("EigenAI FLUX Upscaler Node initialized")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "image": ("IMAGE", {
                    "description": "Input image to upscale"
                }),
                "upscale_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "step": 1,
                    "display": "dropdown",
                    "description": "Upscaling factor"
                }),
                "upscale_method": ("STRING", {
                    "default": "lanczos",
                    "description": "Upscaling method",
                    "display": "dropdown",
                    "choices": ["lanczos", "bicubic", "bilinear", "nearest"]
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_image"
    CATEGORY = "Eigen AI Modular"
    OUTPUT_NODE = False
    
    def upscale_image(self, image, upscale_factor, upscale_method):
        """
        Upscale input image using specified method
        
        Args:
            image (IMAGE): Input image tensor
            upscale_factor (int): Upscaling factor (2, 3, or 4)
            upscale_method (str): Upscaling method
            
        Returns:
            image_tensor (IMAGE): Upscaled image
        """
        try:
            logger.info(f"Starting image upscaling with factor {upscale_factor} using {upscale_method} method")
            
            # Ensure image is a tensor
            if not isinstance(image, torch.Tensor):
                logger.error("Input image is not a tensor")
                return (image,)
            
            # Get original dimensions
            if len(image.shape) == 4:
                batch_size, height, width, channels = image.shape
                image = image[0]  # Take first batch if multiple
            else:
                height, width, channels = image.shape
                batch_size = 1
            
            logger.info(f"Original image dimensions: {height}x{width}x{channels}")
            
            # Calculate new dimensions
            new_height = height * upscale_factor
            new_width = width * upscale_factor
            
            logger.info(f"New dimensions: {new_height}x{new_width}")
            
            # Convert to PIL Image for high-quality upscaling
            if channels == 3:
                # Convert tensor to PIL Image
                image_array = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array)
                
                # Apply upscaling using PIL
                if upscale_method == "lanczos":
                    resampling = Image.Resampling.LANCZOS
                elif upscale_method == "bicubic":
                    resampling = Image.Resampling.BICUBIC
                elif upscale_method == "bilinear":
                    resampling = Image.Resampling.BILINEAR
                elif upscale_method == "nearest":
                    resampling = Image.Resampling.NEAREST
                else:
                    resampling = Image.Resampling.LANCZOS
                
                # Upscale the image
                upscaled_pil = pil_image.resize((new_width, new_height), resampling)
                
                # Convert back to tensor
                upscaled_array = np.array(upscaled_pil).astype(np.float32) / 255.0
                upscaled_tensor = torch.from_numpy(upscaled_array)
                
            else:
                # For non-RGB images, use torch upscaling
                logger.warning(f"Unsupported channel count: {channels}, using torch upscaling")
                
                # Add batch dimension if needed
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                # Use torch upscaling
                upscaled_tensor = F.interpolate(
                    image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                    size=(new_height, new_width),
                    mode=upscale_method,
                    align_corners=False
                ).permute(0, 2, 3, 1)  # BCHW -> BHWC
                
                # Remove batch dimension if it was added
                if batch_size == 1:
                    upscaled_tensor = upscaled_tensor.squeeze(0)
            
            # Ensure tensor is contiguous
            upscaled_tensor = upscaled_tensor.contiguous()
            
            # Add batch dimension for ComfyUI compatibility
            if len(upscaled_tensor.shape) == 3:
                upscaled_tensor = upscaled_tensor.unsqueeze(0)
            
            logger.info(f"Upscaling completed successfully")
            logger.info(f"Final tensor shape: {upscaled_tensor.shape}")
            
            return (upscaled_tensor,)
            
        except Exception as e:
            error_msg = f"Error during upscaling: {str(e)}"
            logger.error(error_msg)
            
            # Return original image on error
            logger.warning("Returning original image due to upscaling error")
            return (image,)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        """
        Force re-execution when upscaling parameters change
        """
        return f"{kwargs.get('upscale_factor', 2)}_{kwargs.get('upscale_method', 'lanczos')}"
