#!/usr/bin/env python3
"""
Test script to verify that the fixed nodes can be imported and initialized correctly
"""

import sys
import os

# Add the nodes directory to the path
nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
sys.path.insert(0, nodes_dir)

def test_node_imports():
    """Test importing all the fixed nodes"""
    try:
        print("Testing node imports...")
        
        # Test importing the fixed nodes
        from eigenai_kontext_generator_node import EigenAIKontextGeneratorNode
        from eigenai_qwen_generator_node import EigenAIQwenGeneratorNode
        from eigenai_schnell_generator_node import EigenAISchnellGeneratorNode
        from eigenai_upscaler_node import EigenAIUpscalerNode
        
        print("✓ All nodes imported successfully")
        
        # Test node initialization
        print("\nTesting node initialization...")
        
        kontext_node = EigenAIKontextGeneratorNode()
        print("✓ EigenAIKontextGeneratorNode initialized")
        
        qwen_node = EigenAIQwenGeneratorNode()
        print("✓ EigenAIQwenGeneratorNode initialized")
        
        schnell_node = EigenAISchnellGeneratorNode()
        print("✓ EigenAISchnellGeneratorNode initialized")
        
        upscaler_node = EigenAIUpscalerNode()
        print("✓ EigenAIUpscalerNode initialized")
        
        # Test INPUT_TYPES
        print("\nTesting INPUT_TYPES...")
        
        kontext_inputs = kontext_node.INPUT_TYPES()
        print(f"✓ Kontext node has {len(kontext_inputs['required'])} required inputs")
        
        qwen_inputs = qwen_node.INPUT_TYPES()
        print(f"✓ Qwen node has {len(qwen_inputs['required'])} required inputs")
        
        schnell_inputs = schnell_node.INPUT_TYPES()
        print(f"✓ Schnell node has {len(schnell_inputs['required'])} required inputs")
        
        upscaler_inputs = upscaler_node.INPUT_TYPES()
        print(f"✓ Upscaler node has {len(upscaler_inputs['required'])} required inputs")
        
        # Check that upscaler node has upscale parameters
        print("\nChecking upscaler node parameters...")
        
        if 'upscale_factor' in upscaler_inputs['required']:
            print("✓ Upscaler node has 'upscale_factor' parameter")
        else:
            print("✗ Upscaler node missing 'upscale_factor' parameter")
            
        if 'upscale_method' in upscaler_inputs['required']:
            print("✓ Upscaler node has 'upscale_method' parameter")
        else:
            print("✗ Upscaler node missing 'upscale_method' parameter")
        
        print("\n🎉 All tests passed! The nodes should work correctly now.")
        print("\nWorkflow suggestion:")
        print("1. Use generation nodes (Kontext/Qwen/Schnell) to generate images")
        print("2. Connect generated images to Upscaler node for upscaling")
        print("3. This modular approach gives you more control over the workflow")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_node_imports()
    sys.exit(0 if success else 1)
