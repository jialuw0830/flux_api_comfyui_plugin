# Eigen AI FLUX API Plugin for ComfyUI

This ComfyUI plugin provides integration with the Eigen AI FLUX API for high-quality content generation and LoRA functionality.

## 🎨 Key Features

- **High-Quality Generation**: Uses FLUX.1-schnell model for high-quality content
- **Multi-LoRA Support**: Supports up to 3 LoRAs simultaneously
- **Large Font Prompt Input**: Dedicated prompt input node with larger, clearer fonts
- **Content Upscaling**: Built-in content upscaling functionality
- **Real-time Status Monitoring**: Monitors API status and system resources

## 🚀 Node Descriptions

### 1. Eigen AI FLUX API Generator (FluxAPINode)

The main generation node that supports:
- **Large Font Prompt Input**: 17px font with multi-line support
- **LoRA Integration**: Up to 3 LoRAs simultaneously
- **Size Control**: 256x256 to 1024x1024
- **Seed Control**: Reproducible generation results
- **Upscaling**: 2x to 4x upscaling options

**Prompt Input Optimization**:
- Uses `display: "textarea"` for larger input areas
- Supports multi-line input up to 2000 characters
- Font size increased from default 12px to 17px
- Better visual feedback and hover effects

### 2. Eigen AI FLUX API Status (FluxAPIModelStatusNode)

Monitors API status and system resources:
- Model loading status
- GPU and VRAM usage
- System memory status
- LoRA loading information

## 🎯 Font Size Comparison

| Input Type | Default Font Size | Optimized Font Size | Improvement |
|------------|-------------------|---------------------|-------------|
| Standard Input | 12px | 16px | +33% |
| Main Generator Prompt | 12px | 17px | +42% |
| Dedicated Prompt Node | 12px | 18px | +50% |

## 🔧 Installation and Usage

### Installation Steps
1. Copy the plugin folder to `ComfyUI/custom_nodes/` directory
2. Restart ComfyUI
3. Search for "Eigen AI FLUX API" in the node browser

### Basic Workflow
1. **Use Dedicated Prompt Node** (Recommended):
   ```
   FluxPromptNode → FluxAPINode → Output
   ```

2. **Direct Input in Main Node**:
   ```
   FluxAPINode → Output
   ```

## 🎨 Style Customization

The plugin includes a custom CSS file (`style.css`) that provides:
- Larger font sizes
- Better visual feedback
- Hover and focus effects
- Custom scrollbar styles
- Responsive design

## 📝 Prompt Writing Tips

### Advantages of Using the Dedicated Prompt Node
- **Larger Fonts**: Easier to read and edit
- **Multi-line Support**: Can write more detailed descriptions
- **Clean Interface**: Focused on prompt input only

### Prompt Examples
```
Main Prompt:
A beautiful landscape painting in Studio Ghibli style, featuring rolling hills, cherry blossoms, and a peaceful atmosphere, soft lighting, detailed textures
```

## 🔗 API Configuration

Default API URL: `http://74.81.65.108:8000`

API URL can be customized through node parameters.

## 📊 Performance Optimization

- Batch processing support
- Smart caching mechanism
- Timeout setting: 5 minutes
- Error handling and retry mechanism

## 🆘 Troubleshooting

### Common Issues
1. **Fonts Still Small**: Ensure CSS file is properly loaded
2. **Nodes Not Showing**: Check plugin installation path
3. **API Connection Failed**: Verify API URL and network connection

### Debug Information
All nodes include detailed logging that can be viewed in the ComfyUI console.

## 📄 License

This plugin follows an open source license. Contributions and improvements are welcome!

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this plugin!

---

**Enjoy using the Eigen AI FLUX API Plugin!** 🎨✨
