"""
ComfyUI Node for Eigen AI FLUX LoRA Management
This node handles LoRA loading and weight management for FLUX API

Key Features:
- LoRA loading and management
- Weight adjustment for up to 3 LoRAs
- LoRA validation and error handling
- Integration with FLUX generation workflow
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EigenAILoraNode:
    """
    ComfyUI Eigen AI FLUX LoRA Management Node
    
    This node manages LoRA loading and weight configuration for FLUX API nodes.
    It focuses solely on LoRA management without handling text processing or image generation.
    
    Usage:
    1. Select LoRA files
    2. Adjust weights for each LoRA
    3. Connect to FLUX generation nodes
    """
    
    def __init__(self):
        logger.info("EigenAI FLUX LoRA Node initialized")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "lora1_name": ("STRING", {
                    "default": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "description": "Primary LoRA file path",
                    "multiline": False
                }),
                "lora1_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Primary LoRA weight"
                }),
            },
            "optional": {
                "lora2_name": ("STRING", {
                    "default": "",
                    "description": "Secondary LoRA file path (optional)",
                    "multiline": False
                }),
                "lora2_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Secondary LoRA weight"
                }),
                "lora3_name": ("STRING", {
                    "default": "",
                    "description": "Tertiary LoRA file path (optional)",
                    "multiline": False
                }),
                "lora3_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Tertiary LoRA weight"
                })
            }
        }
    
    RETURN_TYPES = ("LORA_CONFIG",)
    RETURN_NAMES = ("lora_config",)
    FUNCTION = "process_loras"
    CATEGORY = "Eigen AI Modular"
    OUTPUT_NODE = False
    
    def process_loras(self, lora1_name, lora1_weight, lora2_name="", lora2_weight=1.0, lora3_name="", lora3_weight=1.0):
        """
        Process and validate LoRA configuration
        
        Args:
            lora1_name (str): Primary LoRA file path
            lora1_weight (float): Primary LoRA weight
            lora2_name (str): Secondary LoRA file path (optional)
            lora2_weight (float): Secondary LoRA weight
            lora3_name (str): Tertiary LoRA file path (optional)
            lora3_weight (float): Tertiary LoRA weight
            
        Returns:
            dict: LoRA configuration dictionary
        """
        try:
            lora_config = {
                "loras": [],
                "total_count": 0
            }
            
            # Add primary LoRA (always required)
            if lora1_name and lora1_name.strip():
                lora_config["loras"].append({
                    "name": lora1_name.strip(),
                    "weight": lora1_weight,
                    "type": "primary"
                })
                lora_config["total_count"] += 1
                logger.info(f"Added primary LoRA: {lora1_name.strip()} (weight: {lora1_weight})")
            
            # Add secondary LoRA if specified
            if lora2_name and lora2_name.strip():
                lora_config["loras"].append({
                    "name": lora2_name.strip(),
                    "weight": lora2_weight,
                    "type": "secondary"
                })
                lora_config["total_count"] += 1
                logger.info(f"Added secondary LoRA: {lora2_name.strip()} (weight: {lora2_weight})")
            
            # Add tertiary LoRA if specified
            if lora3_name and lora3_name.strip():
                lora_config["loras"].append({
                    "name": lora3_name.strip(),
                    "weight": lora3_weight,
                    "type": "tertiary"
                })
                lora_config["total_count"] += 1
                logger.info(f"Added tertiary LoRA: {lora3_name.strip()} (weight: {lora3_weight})")
            
            # Validate LoRA configuration
            if lora_config["total_count"] == 0:
                logger.warning("No LoRAs specified, using default configuration")
                # Add default LoRA if none specified
                lora_config["loras"].append({
                    "name": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "weight": 1.0,
                    "type": "default"
                })
                lora_config["total_count"] = 1
            
            # Limit to maximum 3 LoRAs
            if lora_config["total_count"] > 3:
                logger.warning(f"Too many LoRAs specified ({lora_config['total_count']}), limiting to first 3")
                lora_config["loras"] = lora_config["loras"][:3]
                lora_config["total_count"] = 3
            
            logger.info(f"Final LoRA configuration: {lora_config['total_count']} LoRAs loaded")
            
            return (lora_config,)
            
        except Exception as e:
            error_msg = f"Error processing LoRAs: {str(e)}"
            logger.error(error_msg)
            
            # Return default configuration as fallback
            fallback_config = {
                "loras": [{
                    "name": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "weight": 1.0,
                    "type": "fallback"
                }],
                "total_count": 1
            }
            return (fallback_config,)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        """
        Force re-execution when LoRA parameters change
        """
        return f"{kwargs.get('lora1_name', '')}_{kwargs.get('lora1_weight', 1.0)}_{kwargs.get('lora2_name', '')}_{kwargs.get('lora2_weight', 1.0)}_{kwargs.get('lora3_name', '')}_{kwargs.get('lora3_weight', 1.0)}"
