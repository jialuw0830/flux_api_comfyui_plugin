# 🔄 Restart Instructions - View New Prompt Input Node

## Problem Description
If you still only see two nodes in ComfyUI, this is because the newly added `FluxPromptNode` needs a ComfyUI restart to be recognized.

## Solution

### Method 1: Complete ComfyUI Restart (Recommended)
1. **Stop ComfyUI**: Press `Ctrl+C` in the terminal to stop the service
2. **Wait a few seconds**: Ensure all processes are completely stopped
3. **Restart ComfyUI**: Run your startup command
4. **Check nodes**: Search for "Eigen AI FLUX API" in the left node browser

### Method 2: Check Console Output
After restart, check if ComfyUI console displays the following information:
```
✅ Eigen AI FLUX API Plugin loaded successfully!
   - Loaded 3 nodes
   - Available nodes: ['🎨 Eigen AI FLUX API Prompt Input', '🎨 Eigen AI FLUX API Generator', '🤖 Eigen AI FLUX API Status']
```

### Method 3: Verify File Structure
Ensure the following files exist and have correct content:
- `__init__.py` - Contains node registration code
- `nodes/flux_api_node.py` - Contains all three node classes
- `style.css` - Contains font enlargement styles

## Three Nodes You Should See

After restart, you should see on the left:

1. **🎨 Eigen AI FLUX API Prompt Input** (FluxPromptNode)
   - Large font prompt input (18px)
   - Clean and focused interface
   - Multi-line support

2. **🎨 Eigen AI FLUX API Generator** (FluxAPINode)
   - Large font prompt input (17px)
   - Generation parameters
   - LoRA configuration

3. **🤖 Eigen AI FLUX API Status** (FluxAPIModelStatusNode)
   - API status monitoring
   - System resource information

## If You Still Don't See New Nodes

1. **Check file permissions**: Ensure all files have correct read permissions
2. **Check Python path**: Ensure ComfyUI can find the plugin directory
3. **View error logs**: Check ComfyUI console for error messages
4. **Clear cache**: Delete `__pycache__` directories and restart

## Font Enlargement Effects

After restart, you should notice:
- Prompt input box fonts are significantly larger
- Larger input areas with multi-line support
- Better visual styles and hover effects

---

**Important**: Every time you modify plugin code, you need to restart ComfyUI to see the changes!
