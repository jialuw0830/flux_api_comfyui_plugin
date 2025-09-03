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

class EigenAITextNode:
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
                    "description": "Text prompt for generation",
                    "multiline": True,
                    "max_length": 2000,
                    "display": "textarea"
                })
            }
        }
    
    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process_text"
    CATEGORY = "Eigen AI Modular"
    OUTPUT_NODE = False
    
    def process_text(self, prompt):
        """
        Process text prompt for FLUX API
        
        Args:
            prompt (str): Text prompt for generation
            
        Returns:
            str: Processed prompt
        """
        try:
            # Simply return the prompt as is
            final_prompt = prompt.strip()
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
        return f"{kwargs.get('prompt', '')}"
