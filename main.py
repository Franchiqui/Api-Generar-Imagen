from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import httpx
from io import BytesIO
from PIL import Image
import uuid

app = FastAPI()

# --- CONFIGURACIÓN ---
POCKETBASE_URL = "https://zeus-media-studio-ia.fly.dev" 
POCKETBASE_COLLECTION = "imagen_generada"
# URL de Hugging Face para FLUX.1-schnell (puedes cambiarlo por /FLUX.1-dev si prefieres)
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
    return {"Hello": "World"}

@app.post("/test_generate_and_proxy/")
async def test_generate_and_proxy(
    api_key: str = Form(...), # API Key de OpenAI o Access Token de Hugging Face
    model: str = Form("dall-e-3"), # "dall-e-2", "dall-e-3" o "flux"
    prompt: str = Form(...),
    n: int = Form(1),
    size: str = Form("1024x1024")
):
    try:
        image_bytes_list = []

        # --- LÓGICA SEGÚN EL MODELO ---
        if "flux" in model.lower():
            # Hugging Face Inference API
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(FLUX_API_URL, headers=headers, json={"inputs": prompt})
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Hugging Face Error: {response.text}")
                image_bytes_list.append(response.content)
        else:
            # OpenAI API (DALL-E)
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
            
            # Descargar imágenes de OpenAI para subirlas a PocketBase
            async with httpx.AsyncClient() as client:
                for data in response.data:
                    img_res = await client.get(data.url)
                    if img_res.status_code == 200:
                        image_bytes_list.append(img_res.content)
                    else:
                        raise HTTPException(status_code=img_res.status_code, detail="Error fetching image from OpenAI")

        # --- SUBIDA COMÚN A POCKETBASE ---
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
    uvicorn.run(app, host="0.0.0.0", port=5081)
