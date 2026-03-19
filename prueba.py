import requests
import json
import time
from PIL import Image
from io import BytesIO

# --- 1. ENVIAR WORKFLOW ---
workflow = {
    "1": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        }
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_3_4b.safetensors",
            "type": "flux2"
        }
    },
    "3": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "flux-2-klein-4b.safetensors",
            "weight_dtype": "fp8_e4m3fn"
        }
    },
    "4": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "flux2-vae.safetensors"
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "A photo of a woman's face",
            "clip": ["2", 0]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["2", 0]
        }
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["3", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["1", 0]
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["4", 0]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "test_flux",
            "images": ["8", 0]
        }
    }
}

print("🚀 Enviando workflow a ComfyUI...")
response = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow})
result = response.json()
prompt_id = result["prompt_id"]
print(f"✅ Prompt ID: {prompt_id}")

# --- 2. ESPERAR A QUE TERMINE ---
print("⏳ Esperando a que ComfyUI procese...")
max_attempts = 30
for attempt in range(max_attempts):
    time.sleep(2)  # Esperar 2 segundos entre intentos
    print(f"   Intento {attempt + 1}/{max_attempts}")
    
    history_response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
    if history_response.status_code == 200:
        history = history_response.json()
        if prompt_id in history:
            print("✅ Procesamiento completado!")
            outputs = history[prompt_id]["outputs"]
            
            # Buscar la imagen generada
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    image_data = node_output["images"][0]
                    print(f"📸 Imagen encontrada: {image_data['filename']}")
                    
                    # Descargar la imagen
                    filename = image_data["filename"]
                    subfolder = image_data["subfolder"]
                    img_response = requests.get(
                        f"http://127.0.0.1:8188/view",
                        params={
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": image_data["type"]
                        }
                    )
                    
                    if img_response.status_code == 200:
                        # Guardar la imagen
                        with open(f"imagen_generada_{int(time.time())}.png", "wb") as f:
                            f.write(img_response.content)
                        print(f"💾 Imagen guardada!")
                        
                        # Mostrar info de la imagen
                        img = Image.open(BytesIO(img_response.content))
                        print(f"📏 Tamaño: {img.width}x{img.height}")
                        
                        # Mostrar dónde está
                        import os
                        print(f"📍 Ruta: {os.path.abspath(f'imagen_generada_{int(time.time())}.png')}")
                    break
            break
else:
    print("❌ Timeout: ComfyUI no completó el procesamiento")