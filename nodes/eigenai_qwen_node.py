"""
ComfyUI Node for Qwen-compatible Image API (Eigen AI Flux API on :8010)

This node targets the UI/API hosted at `http://74.81.65.108:8010/ui` and its
`POST /generate` + `GET /download/{filename}` endpoints. It mirrors the style of
`FluxAPINode`, supports up to 3 LoRAs, and adds `guidance_scale`.
"""

import json
import requests
import numpy as np
from PIL import Image
import io
import logging
import torch


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EigenAIQwenNode:
    """
    ComfyUI node integrating the :8010 image generation API.

    Usage:
    1. 填写 prompt 与尺寸
    2. 设置 guidance_scale、可选 seed
    3. 最多选择 3 个 LoRA 与权重
    4. 可选开启超分 upscale 与倍率 upscale_factor
    5. 运行生成并自动下载返回图片
    """

    def __init__(self):
        self.api_base_url = "http://74.81.65.108:8010"
        self.session = requests.Session()
        self.session.timeout = 300

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "description": "Text prompt from text node (connect from EigenAITextNode)"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number"
                }),
                "upscale": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable Remacri ESRGAN upscaling"
                }),
                "upscale_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "step": 2,
                    "display": "dropdown"
                }),
                "api_url": ("STRING", {
                    "default": "http://74.81.65.108:8010",
                    "description": "Qwen-compatible image API base URL"
                }),
                "lora1_name": ("STRING", {
                    "default": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                    "description": "Primary LoRA name/path (use empty to disable)",
                    "multiline": False
                }),
                "lora1_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "lora2_name": ("STRING", {
                    "default": "none",
                    "description": "Second LoRA name (optional, use 'none' to disable)",
                    "multiline": False
                }),
                "lora2_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "lora3_name": ("STRING", {
                    "default": "none",
                    "description": "Third LoRA name (optional, use 'none' to disable)",
                    "multiline": False
                }),
                "lora3_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Eigen AI FLUX API"
    OUTPUT_NODE = False

    def generate_image(self, prompt, width, height, guidance_scale, seed, upscale, upscale_factor, api_url,
                       lora1_name="/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors", lora1_weight=1.0,
                       lora2_name="none", lora2_weight=1.0,
                       lora3_name="none", lora3_weight=1.0):
        try:
            if api_url and api_url != "http://74.81.65.108:8010":
                self.api_base_url = api_url

            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
            }

            if seed != -1:
                payload["seed"] = seed

            if upscale:
                payload["upscale"] = True
                payload["upscale_factor"] = upscale_factor

            # Build LoRA configuration (up to 3)
            loras_to_apply = []
            # Always include LoRA 1 (default provided like flux_api_node)
            if lora1_name and str(lora1_name).strip():
                loras_to_apply.append({
                    "name": str(lora1_name).strip(),
                    "weight": float(lora1_weight)
                })
                logger.info(f"Added LoRA 1: {str(lora1_name).strip()} (weight: {lora1_weight})")

            # Add LoRA 2 if valid
            if (lora2_name and str(lora2_name).strip() and str(lora2_name).strip().lower() != "none"):
                loras_to_apply.append({
                    "name": str(lora2_name).strip(),
                    "weight": float(lora2_weight)
                })
                logger.info(f"Added LoRA 2: {str(lora2_name).strip()} (weight: {lora2_weight})")

            # Add LoRA 3 if valid
            if (lora3_name and str(lora3_name).strip() and str(lora3_name).strip().lower() != "none"):
                loras_to_apply.append({
                    "name": str(lora3_name).strip(),
                    "weight": float(lora3_weight)
                })
                logger.info(f"Added LoRA 3: {str(lora3_name).strip()} (weight: {lora3_weight})")

            if len(loras_to_apply) > 3:
                loras_to_apply = loras_to_apply[:3]
                logger.warning("Too many LoRAs specified, limited to first 3")

            payload["loras"] = loras_to_apply

            logger.info(f"Sending request to Qwen API: {self.api_base_url}/generate")
            logger.info(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

            response = self.session.post(
                f"{self.api_base_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} {response.text}")

            result = response.json()

            download_url = result.get("download_url", "")
            if not download_url:
                raise Exception("No download URL in API response")

            if download_url.startswith("/"):
                full_download_url = f"{self.api_base_url}{download_url}"
            else:
                full_download_url = f"{self.api_base_url}/{download_url}"

            logger.info(f"Downloading image from: {full_download_url}")
            image_response = self.session.get(full_download_url, timeout=60)
            if image_response.status_code != 200:
                raise Exception(f"Failed to download image: {image_response.status_code}")

            image_data = image_response.content
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            image_array = np.array(pil_image).astype(np.float32) / 255.0
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, 0)

            image_tensor = torch.from_numpy(image_array).contiguous()
            return (image_tensor,)

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            placeholder = np.zeros((height, width, 3), dtype=np.float32)
            placeholder[:, :, 0] = 0.8
            placeholder = np.expand_dims(placeholder, 0)
            return (torch.from_numpy(placeholder),)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "EigenAIQwenNode": EigenAIQwenNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "EigenAIQwenNode": "Eigen AI Qwen API Generator"
}


