"""
FLUX API Plugin for ComfyUI
A plugin that integrates FLUX.1-schnell model with LoRA support into ComfyUI
"""

import os
import sys

# Add the nodes directory to Python path
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all nodes from the nodes directory
nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
if os.path.exists(nodes_dir):
    sys.path.append(nodes_dir)
    
    # Import the main nodes
    try:
        from flux_api_node import FluxAPINode
        from kontext_api_node import KontextAPINode
        from qwen_api_node import QwenAPINode
        
        # Register the nodes
        NODE_CLASS_MAPPINGS.update({
            "FluxAPINode": FluxAPINode,
            "KontextAPINode": KontextAPINode,
            "QwenAPINode": QwenAPINode
        })
        
        NODE_DISPLAY_NAME_MAPPINGS.update({
            "FluxAPINode": "Eigen AI FLUX Schnell API Generator",
            "KontextAPINode": "Eigen AI FLUX Kontext API Generator",
            "QwenAPINode": "Eigen AI Qwen API Generator"
        })
        
        print("Eigen AI FLUX API Plugin loaded successfully!")
        print(f"   - Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
        print(f"   - Available nodes: {list(NODE_DISPLAY_NAME_MAPPINGS.values())}")
        print("Websocket connected")
        
    except ImportError as e:
        print(f"Failed to load FLUX API Plugin: {e}")
        print("   Please check that all dependencies are installed:")
        print("   pip install requests pillow numpy torch")
else:
    print("FLUX API Plugin: nodes directory not found")

# Plugin metadata
__version__ = "1.0.0"
__author__ = "Eigen AI"
__description__ = "Eigen AI FLUX API integration for ComfyUI with LoRA support, large font prompt inputs, and Kontext image-to-image generation"
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
        "name": "Eigen AI FLUX API",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "required_packages": REQUIRED_PACKAGES
    }
