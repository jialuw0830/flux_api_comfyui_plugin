# Eigen AI FLUX API Plugin for ComfyUI

A **ComfyUI plugin** that integrates with the **Eigen AI FLUX API**, providing high-quality image generation with **LoRA** support.

---

## 🚀 Quick Installation

### 1. Install ComfyUI

If you haven’t installed **ComfyUI** yet:

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# (Optional) Start ComfyUI
python main.py
```

---

### 2. Install the Plugin

#### Method A: Install via ComfyUI-Manager (Recommended)

1. Install **ComfyUI-Manager** if not already installed:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   ```

   > Restart ComfyUI after installation.

2. Install this plugin via Manager:

   1. Start ComfyUI:

      ```bash
      python main.py
      ```
   2. Open **ComfyUI-Manager**
   3. Search for **“Eigen AI FLUX API”** or **“flux api”**
   4. Click **Install** and wait until it finishes
   5. Restart ComfyUI
   6. Verify nodes appear in the left panel by searching **“Eigen AI FLUX API”**

---

#### Method B: Manual Install

If you prefer manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jialuw0830/flux_api_comfyui_plugin.git

# (Optional) Start ComfyUI
cd ComfyUI
python main.py
```

---

### 3. Post-Installation Steps

1. Restart **ComfyUI**
2. Search for **“Eigen AI FLUX API”** in the node browser
3. Generated images will be saved in:

   ```
   ComfyUI/output
   ```

---

## 🎨 Key Features

* **High-Quality Generation** — Uses **FLUX.1-schnell** model
* **Multi-LoRA Support** — Up to **3 LoRAs simultaneously**
* **Upscaling** — Supports **2×** and **4×** upscaling
* **Fast Performance** — Uses **quantized models** for quick generation

---

## 🔧 Node Descriptions

### 1. Eigen AI FLUX API Generator

Main generation node supporting:

* Up to **3 LoRAs**
* **Seed control** for reproducibility
* **2× / 4× upscaling**

### 2. Eigen AI FLUX API Status

Monitors **API status** and **system resources**

---

## 📝 Usage Examples

### Basic Workflow

```
FluxPromptNode → FluxAPINode → Output
```

### Prompt Example

```
A beautiful landscape painting in Studio Ghibli style, 
featuring rolling hills, cherry blossoms, and a peaceful atmosphere, 
soft lighting, detailed textures
```

---

## 🔗 API Configuration

* **Default API URL**:

  ```
  http://74.81.65.108:8000
  ```
* API URL can be customized in **node parameters**.

---

## 🆘 Troubleshooting

* **Error: "This action is not allowed with this security level configuration."**
  Solution:
  Edit:

  ```
  ComfyUI/user/default/ComfyUI-Manager/config.ini
  ```

  Change:

  ```
  security_level = normal
  ```

  to:

  ```
  security_level = weak
  ```

  Then restart **ComfyUI**.

---

## 📄 License

**MIT License**

---

✨ Enjoy using the **Eigen AI FLUX API Plugin**! 🎨

---

