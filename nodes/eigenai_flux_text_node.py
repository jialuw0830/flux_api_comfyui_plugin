"""
ComfyUI Node for Eigen AI FLUX Text Processing
This node handles text prompt processing and preparation for FLUX API

Key Features:
- Text prompt input and validation
- Prompt enhancement and formatting
- Text concatenation support
- Integration with FLUX API workflow
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EigenAIFluxTextNode:
    """
    ComfyUI Eigen AI FLUX Text Processing Node
    
    This node processes and prepares text prompts for use with FLUX API nodes.
    It focuses solely on text processing without handling image generation or LoRA management.
    
    Usage:
    1. Input your text prompt
    2. Optionally add style modifiers
    3. Connect to FLUX generation nodes
    """
    
    def __init__(self):
        logger.info("EigenAI FLUX Text Node initialized")
        
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        """
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A beautiful landscape painting in Studio Ghibli style",
                    "description": "Main text prompt for generation",
                    "multiline": True,
                    "max_length": 2000,
                    "display": "textarea"
                }),
                "style_modifier": ("STRING", {
                    "default": "",
                    "description": "Additional style modifiers (optional)",
                    "multiline": True,
                    "max_length": 1000,
                    "display": "textarea"
                }),
                "quality_tags": ("STRING", {
                    "default": "masterpiece, best quality, highly detailed",
                    "description": "Quality enhancement tags",
                    "multiline": True,
                    "max_length": 500,
                    "display": "textarea"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process_text"
    CATEGORY = "Eigen AI FLUX"
    OUTPUT_NODE = False
    
    def process_text(self, prompt, style_modifier, quality_tags):
        """
        Process and combine text inputs for FLUX API
        
        Args:
            prompt (str): Main text prompt
            style_modifier (str): Style modifiers
            quality_tags (str): Quality enhancement tags
            
        Returns:
            str: Combined prompt
        """
        try:
            # Combine main prompt with style modifier
            combined_prompt = prompt.strip()
            if style_modifier and style_modifier.strip():
                combined_prompt += f", {style_modifier.strip()}"
            
            # Add quality tags
            if quality_tags and quality_tags.strip():
                combined_prompt += f", {quality_tags.strip()}"
            
            # Clean up the final prompt
            final_prompt = combined_prompt.strip()
            if final_prompt.endswith(","):
                final_prompt = final_prompt[:-1]
            
            logger.info(f"Processed prompt: {final_prompt}")
            
            return (final_prompt,)
            
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            logger.error(error_msg)
            
            # Return original prompt as fallback
            return (prompt,)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        """
        Force re-execution when text parameters change
        """
        return f"{kwargs.get('prompt', '')}_{kwargs.get('style_modifier', '')}_{kwargs.get('quality_tags', '')}"
