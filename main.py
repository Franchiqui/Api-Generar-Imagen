from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import httpx
from io import BytesIO
from PIL import Image
import uuid
import base64
import aiohttp
import asyncio
import os

app = FastAPI()

# --- CONFIGURACIÓN ---
POCKETBASE_URL = "https://zeus-media-studio-ia.fly.dev" 
POCKETBASE_COLLECTION = "imagen_generada"
FLUX_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# ---------------------

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "API funcionando", "endpoints": ["/test_generate_and_proxy/", "/edit_with_comfyui_flux/", "/list_images/", "/delete_image/{image_id}"]}

@app.post("/test_generate_and_proxy/")
async def test_generate_and_proxy(
    api_key: str = Form(...),
    model: str = Form("dall-e-3"),
    prompt: str = Form(...),
    n: int = Form(1),
    size: str = Form("1024x1024")
):
    try:
        image_bytes_list = []

        if "flux" in model.lower():
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(FLUX_API_URL, headers=headers, json={"inputs": prompt})
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Hugging Face Error: {response.text}")
                image_bytes_list.append(response.content)
        else:
            client_openai = OpenAI(api_key=api_key)
            actual_n = 1 if model == "dall-e-3" else n
            
            response = client_openai.images.generate(
                model=model,
                prompt=prompt,
                n=actual_n,
                size=size
            )
            
            if not response.data:
                raise HTTPException(status_code=400, detail="Failed to generate images with OpenAI.")
            
            async with httpx.AsyncClient() as client:
                for data in response.data:
                    img_res = await client.get(data.url)
                    if img_res.status_code == 200:
                        image_bytes_list.append(img_res.content)
                    else:
                        raise HTTPException(status_code=img_res.status_code, detail="Error fetching image from OpenAI")

        final_image_urls = []
        if not image_bytes_list:
            raise HTTPException(status_code=404, detail="No images were generated")

        async with httpx.AsyncClient() as client:
            for content in image_bytes_list:
                image = Image.open(BytesIO(content))
                img_io = BytesIO()
                image.save(img_io, format='JPEG')
                img_io.seek(0)

                files = {'imagen': ('generated_image.jpg', img_io, 'image/jpeg')}
                upload_url = f"{POCKETBASE_URL}/api/collections/{POCKETBASE_COLLECTION}/records"
                
                upload_res = await client.post(upload_url, files=files)
                if upload_res.status_code != 200:
                    raise HTTPException(status_code=upload_res.status_code, detail=f"PocketBase upload error: {upload_res.text}")

                res_json = upload_res.json()
                record_id = res_json['id']
                filename = res_json['imagen']
                coll_id = res_json['collectionId']

                pb_url = f"{POCKETBASE_URL}/api/files/{coll_id}/{record_id}/{filename}"
                final_image_urls.append({"id": record_id, "url": pb_url})

        return {"images": final_image_urls}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== NUEVO ENDPOINT PARA FLUX.2 KLEIN =====
@app.post("/edit_with_comfyui_flux/")
async def edit_with_comfyui_flux(
    prompt: str = Form(...),
    image_file: UploadFile = File(...),
    mask_file: UploadFile = File(None),
    strength: float = Form(0.8),
    seed: int = Form(42),
    steps: int = Form(4),
    cfg: float = Form(1.0),
    use_base_model: bool = Form(False),
    comfyui_url: str = Form("http://127.0.0.1:8188"),
):
    """
    Edita una imagen usando FLUX.2 Klein con la estructura correcta de carpetas
    """
    try:
        # --- 1. VERIFICAR CONEXIÓN ---
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{comfyui_url}/", timeout=5) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="ComfyUI no responde")
                    print("✅ ComfyUI conectado")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"No se puede conectar a ComfyUI: {str(e)}")
        
        # --- 2. SELECCIONAR MODELO (desde unet/) ---
        if use_base_model:
            model_name = "flux-2-klein-base-4b.safetensors"
            if steps == 4:
                steps = 30
            if cfg == 1.0:
                cfg = 3.5
        else:
            model_name = "flux-2-klein-4b.safetensors"
            if steps > 4:
                steps = 4
            if cfg != 1.0:
                cfg = 1.0
        
        print(f"📌 Usando modelo UNET: {model_name}")
        print(f"📌 Parámetros: steps={steps}, cfg={cfg}, strength={strength}")
        
        # --- 3. PREPARAR IMAGEN ---
        image_content = await image_file.read()
        if len(image_content) == 0:
            raise HTTPException(status_code=400, detail="La imagen está vacía")
        
        temp_dir = "C:\\temp\\comfyui_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_image_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.png")
        
        try:
            image = Image.open(BytesIO(image_content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            width = (image.width // 16) * 16
            height = (image.height // 16) * 16
            if width != image.width or height != image.height:
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            print(f"📏 Imagen: {image.width}x{image.height}")
            image.save(temp_image_path, format="PNG")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar imagen: {str(e)}")
        
        # --- 4. PREPARAR MÁSCARA ---
        mask_path = None
        if mask_file:
            mask_content = await mask_file.read()
            if len(mask_content) > 0:
                try:
                    mask = Image.open(BytesIO(mask_content))
                    mask = mask.convert('L')
                    if mask.size != (image.width, image.height):
                        mask = mask.resize((image.width, image.height), Image.Resampling.LANCZOS)
                    
                    mask_path = os.path.join(temp_dir, f"mask_{uuid.uuid4()}.png")
                    mask.save(mask_path, format="PNG")
                    print("✅ Máscara cargada")
                except Exception as e:
                    print(f"⚠️ Error con máscara: {e}")
        
        # --- 5. WORKFLOW CORREGIDO ---
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": temp_image_path
                }
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_4b.safetensors",
                    "type": "flux2"  # ¡IMPORTANTE: flux2, no flux!
                }
            },
            "3": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": model_name,  # Ahora buscará en unet/
                    "weight_dtype": "fp8_e4m3fn"
                }
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "flux2-vae.safetensors"  # Buscará en vae/
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
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
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 0]
                }
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": strength,
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0]
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["4", 0]
                }
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "flux_output",
                    "images": ["9", 0]
                }
            }
        }
        
        if mask_path:
            workflow["11"] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": mask_path
                }
            }
            workflow["12"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 0],
                    "mask": ["11", 0],
                    "grow_mask_by": 6
                }
            }
            workflow["8"]["inputs"]["latent_image"] = ["12", 0]
        
        # --- 6. ENVIAR A COMFYUI ---
        async with aiohttp.ClientSession() as session:
            print("🚀 Enviando a ComfyUI...")
            
            async with session.post(
                f"{comfyui_url}/prompt",
                json={"prompt": workflow},
                timeout=30
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ Error: {error_text}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error de ComfyUI: {error_text[:500]}"
                    )
                
                result = await resp.json()
                prompt_id = result["prompt_id"]
                print(f"✅ Prompt ID: {prompt_id}")
            
            # --- 7. ESPERAR RESULTADO ---
            max_attempts = 60
            edited_image_bytes = None
            
            for attempt in range(max_attempts):
                print(f"⏳ Intento {attempt+1}/{max_attempts}")
                await asyncio.sleep(2)
                
                async with session.get(f"{comfyui_url}/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if prompt_id in history:
                            outputs = history[prompt_id]["outputs"]
                            
                            for node_output in outputs.values():
                                if "images" in node_output:
                                    img_data = node_output["images"][0]
                                    
                                    async with session.get(
                                        f"{comfyui_url}/view",
                                        params={
                                            "filename": img_data["filename"],
                                            "subfolder": img_data["subfolder"],
                                            "type": img_data["type"]
                                        }
                                    ) as img_resp:
                                        if img_resp.status == 200:
                                            edited_image_bytes = await img_resp.read()
                                            print("✅ Imagen generada")
                                            break
                            if edited_image_bytes:
                                break
            
            if not edited_image_bytes:
                raise HTTPException(status_code=504, detail="Timeout: ComfyUI no generó la imagen")
        
        # --- 8. LIMPIAR Y SUBIR ---
        try:
            os.remove(temp_image_path)
            if mask_path:
                os.remove(mask_path)
        except:
            pass
        
        edited_image = Image.open(BytesIO(edited_image_bytes))
        img_io = BytesIO()
        edited_image.save(img_io, format='JPEG', quality=95)
        img_io.seek(0)
        
        async with httpx.AsyncClient() as client_pb:
            files_pb = {'imagen': ('edited_image.jpg', img_io, 'image/jpeg')}
            upload_res = await client_pb.post(
                f"{POCKETBASE_URL}/api/collections/{POCKETBASE_COLLECTION}/records",
                files=files_pb
            )
            
            if upload_res.status_code != 200:
                raise HTTPException(
                    status_code=upload_res.status_code,
                    detail=f"Error subiendo a PocketBase: {upload_res.text}"
                )
            
            res_json = upload_res.json()
            pb_url = f"{POCKETBASE_URL}/api/files/{res_json['collectionId']}/{res_json['id']}/{res_json['imagen']}"
        
        return {
            "success": True,
            "message": "Imagen editada exitosamente",
            "edited_image": {
                "id": res_json['id'],
                "url": pb_url
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/list_images/")
async def list_images():
    try:
        async with httpx.AsyncClient() as client:
            url = f"{POCKETBASE_URL}/api/collections/{POCKETBASE_COLLECTION}/records"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                image_urls = []
                for item in items:
                    image_urls.append({
                        "id": item['id'],
                        "url": f"{POCKETBASE_URL}/api/files/{item['collectionId']}/{item['id']}/{item['imagen']}"
                    })
                return JSONResponse(content=image_urls)
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch images from PocketBase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_image/{image_id}")
async def delete_image(image_id: str):
    try:
        async with httpx.AsyncClient() as client:
            url = f"{POCKETBASE_URL}/api/collections/{POCKETBASE_COLLECTION}/records/{image_id}"
            response = await client.delete(url)
            if response.status_code == 204:
                return {"detail": "Image deleted successfully"}
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to delete image from PocketBase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 API iniciando en http://localhost:5081")
    print("📌 Endpoints disponibles:")
    print("   - GET  /")
    print("   - POST /test_generate_and_proxy/")
    print("   - POST /edit_with_comfyui_flux/")
    print("   - GET  /list_images/")
    print("   - DELETE /delete_image/{image_id}")
    uvicorn.run(app, host="0.0.0.0", port=5081)
