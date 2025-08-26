 Eigen AI FLUX API Plugin for ComfyUI

A ComfyUI plugin that integrates with the Eigen AI FLUX API, providing high-quality content generation and LoRA functionality.

 🚀 Quick Installation

 Install ComfyUI

If you haven't installed ComfyUI yet, please install it first:

```bash
 Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

 Install dependencies
pip install -r requirements.txt

 Start ComfyUI (optional)
python main.py
```

 Method A: Install via ComfyUI-Manager (recommended)


If you don't have ComfyUI-Manager installed:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
 Restart ComfyUI after installation
```

Then install this plugin via Manager:

1. Start ComfyUI: `python main.py`
2. Open `ComfyUI-Manager`
3. In the search box, type “Eigen AI FLUX API” or “flux api”
4. Click Install and wait until it finishes
5. Restart ComfyUI and search “Eigen AI FLUX API” in the left node panel to verify the nodes appear

 Method B: Manual Install

If you don't have ComfyUI-Manager installed:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
 Restart ComfyUI after installation
```


```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jialuw0830/flux_api_comfyui_plugin.git

 Start ComfyUI (optional)
cd ComfyUI
python main.py
```


 Post-Installation Steps

1. Restart ComfyUI
2. Search for "Eigen AI FLUX API" in the node browser
3. Generated images will be saved in `ComfyUI/output`

 🎨 Key Features

- **High-Quality Generation**: Uses FLUX.1-schnell model
- **Multi-LoRA Support**: Supports up to 3 LoRAs simultaneously
- **Upscaling**: Supports 2x and 4x upscaling
- **Fast Performance**: Uses quantized models for quick generation

 🔧 Node Descriptions

 1. Eigen AI FLUX API Generator
Main generation node that supports:

- Up to 3 LoRAs simultaneously
- Seed control for reproducible results
- 2x/4x upscaling options

 2. Eigen AI FLUX API Status
Monitors API status and system resources

 📝 Usage Examples

 Basic Workflow
```
FluxPromptNode → FluxAPINode → Output
```

 Prompt Example
```
A beautiful landscape painting in Studio Ghibli style, featuring rolling hills, cherry blossoms, and a peaceful atmosphere, soft lighting, detailed textures
```

 🔗 API Configuration

Default API URL: `http://74.81.65.108:8000`

API URL can be customized through node parameters.

 🆘 Troubleshooting

 **This action is not allowed with this security level configuration.**: Edit `ComfyUI/user/default/ComfyUI-Manager/config.ini`, change `security_level = normal` to `security_level = weak`, then restart ComfyUI


 📄 License

MIT License

---

**Enjoy using the Eigen AI FLUX API Plugin!** 🎨✨
