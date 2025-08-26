# Eigen AI FLUX API Plugin for ComfyUI

A ComfyUI plugin that integrates with the Eigen AI FLUX API, providing high-quality content generation and LoRA functionality.

## 🚀 Quick Installation

### Step 1: Install ComfyUI

If you haven't installed ComfyUI yet, please install it first:

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Start ComfyUI (optional)
python main.py
```

### Step 2: Install FLUX API Plugin

#### Method 1: Git Clone (Recommended，make sure you have installed ComfyUI)

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jialuw0830/flux_api_comfyui_plugin.git

# Start ComfyUI (optional)
cd ComfyUI
python main.py
```

#### Method 2: Manual Download

1. Download the ZIP file and extract it to the `ComfyUI/custom_nodes/` directory
2. Rename the folder to `flux_api_comfyui_plugin`

### Post-Installation Steps

1. Restart ComfyUI
2. Search for "Eigen AI FLUX API" in the node browser
3. Generated images will be saved in `ComfyUI/output`

## 🎨 Key Features

- **High-Quality Generation**: Uses FLUX.1-schnell model
- **Multi-LoRA Support**: Supports up to 3 LoRAs simultaneously
- **Upscaling**: Supports 2x and 4x upscaling
- **Fast Performance**: Uses quantized models for quick generation

## 🔧 Node Descriptions

### 1. Eigen AI FLUX API Generator
Main generation node that supports:

- Up to 3 LoRAs simultaneously
- Seed control for reproducible results
- 2x/4x upscaling options

### 2. Eigen AI FLUX API Status
Monitors API status and system resources

## 📝 Usage Examples

### Basic Workflow
```
FluxPromptNode → FluxAPINode → Output
```

### Prompt Example
```
A beautiful landscape painting in Studio Ghibli style, featuring rolling hills, cherry blossoms, and a peaceful atmosphere, soft lighting, detailed textures
```

## 🔗 API Configuration

Default API URL: `http://74.81.65.108:8000`

API URL can be customized through node parameters.

## 🆘 Troubleshooting

1. **Nodes Not Showing**: Check if the installation path is correct
2. **API Connection Failed**: Verify API URL and network connection
3. **Font Still Small**: Ensure CSS file is properly loaded

## 📄 License

MIT License

---

**Enjoy using the Eigen AI FLUX API Plugin!** 🎨✨
