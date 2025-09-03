"""
ComfyUI Plugin for Eigen AI FLUX API Integration
This plugin provides both unified and modular nodes for FLUX API

Features:
- Unified nodes: Direct API integration
- Modular nodes: Separated functionality for better workflow control
"""

import os
import sys

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all nodes from the nodes directory
nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
if os.path.exists(nodes_dir):
    sys.path.append(nodes_dir)
    # Import the main unified nodes (only from nodes root, no subfolders)
    try:
        from eigenai_schnell_node import EigenAISchnellNode
        from eigenai_kontext_node import EigenAIKontextNode
        from eigenai_qwen_node import EigenAIQwenNode
        
        # Register the main unified nodes
        NODE_CLASS_MAPPINGS.update({
            "EigenAISchnellNode": EigenAISchnellNode,
            "EigenAIKontextNode": EigenAIKontextNode,
            "EigenAIQwenNode": EigenAIQwenNode
        })
        
        NODE_DISPLAY_NAME_MAPPINGS.update({
            "EigenAISchnellNode": "Eigen AI Schnell Generator",
            "EigenAIKontextNode": "Eigen AI Kontext Generator",
            "EigenAIQwenNode": "Eigen AI Qwen Generator"
        })
        
        print("Eigen AI Unified API Plugin loaded successfully!")
        print(f"   - Loaded {len(NODE_CLASS_MAPPINGS)} unified nodes")
        print(f"   - Available nodes: {list(NODE_DISPLAY_NAME_MAPPINGS.values())}")
        
    except ImportError as e:
        print(f"Failed to load main unified API nodes: {e}")
    
    # Import the new modular nodes (only from nodes root, no subfolders)
    try:
        from eigenai_text_node import EigenAITextNode
        from eigenai_lora_node import EigenAILoraNode
        from eigenai_schnell_generator_node import EigenAISchnellGeneratorNode
        from eigenai_kontext_generator_node import EigenAIKontextGeneratorNode
        from eigenai_qwen_generator_node import EigenAIQwenGeneratorNode
        from eigenai_upscaler_node import EigenAIUpscalerNode
        
        # Register the new modular nodes
        NODE_CLASS_MAPPINGS.update({
            "EigenAITextNode": EigenAITextNode,
            "EigenAILoraNode": EigenAILoraNode,
            "EigenAISchnellGeneratorNode": EigenAISchnellGeneratorNode,
            "EigenAIKontextGeneratorNode": EigenAIKontextGeneratorNode,
            "EigenAIQwenGeneratorNode": EigenAIQwenGeneratorNode,
            "EigenAIUpscalerNode": EigenAIUpscalerNode
        })
        
        NODE_DISPLAY_NAME_MAPPINGS.update({
            "EigenAITextNode": "EigenAI Text Processor",
            "EigenAILoraNode": "EigenAI LoRA Manager",
            "EigenAISchnellGeneratorNode": "EigenAI Schnell Generator",
            "EigenAIKontextGeneratorNode": "EigenAI Kontext Generator",
            "EigenAIQwenGeneratorNode": "EigenAI Qwen Generator",
            "EigenAIUpscalerNode": "EigenAI Upscaler"
        })
        
        print(f"   - Loaded {len(NODE_CLASS_MAPPINGS)} total nodes (including modular nodes)")
        print(f"   - All available nodes: {list(NODE_DISPLAY_NAME_MAPPINGS.values())}")
        print("Websocket connected")
        
    except ImportError as e:
        print(f"Failed to load modular API nodes: {e}")
        print("   Please check that all modular node files are present in the nodes directory")
else:
    print("Eigen AI API Plugin: nodes directory not found")

# Plugin metadata
__version__ = "2.0.0"
__author__ = "Eigen AI"
__description__ = "Modular Eigen AI API integration for ComfyUI"
__url__ = "https://github.com/eigenai/flux_api_plugin"

# ComfyUI plugin requirements
REQUIRED_PACKAGES = [
    "requests>=2.25.0",
    "pillow>=8.0.0", 
    "numpy>=1.19.0",
    "torch>=1.8.0"
]

def get_required_packages():
    """Return list of required packages for ComfyUI plugin manager"""
    return REQUIRED_PACKAGES

def get_plugin_info():
    """Return plugin information for ComfyUI"""
    return {
        "name": "Eigen AI API",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "required_packages": REQUIRED_PACKAGES
    }
