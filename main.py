from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import httpx
from io import BytesIO
from PIL import Image
import uuid

app = FastAPI()

# --- CONFIGURACIÓN DE POCKETBASE ---
# Puedes cambiar esta URL por la de tu instancia de PocketBase
POCKETBASE_URL = "https://zeus-media-studio-ia.fly.dev" 
POCKETBASE_COLLECTION = "imagen_generada"
# ----------------------------------

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
    api_key: str = Form(...),
    model: str = Form("dall-e-3"),
    prompt: str = Form(...),
    n: int = Form(1),
    size: str = Form("1024x1024")
):
    try:
        client_openai = OpenAI(api_key=api_key)
        
        actual_n = n
        if model == "dall-e-3":
            actual_n = 1

        response = client_openai.images.generate(
            model=model,
            prompt=prompt,
            n=actual_n,
            size=size
        )
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to generate images.")
        
        images = [data.url for data in response.data]
        image_urls = []
        
        async with httpx.AsyncClient() as client:
            for image_url in images:
                # Descargar la imagen de OpenAI
                proxy_response = await client.get(image_url)
                if proxy_response.status_code == 200:
                    content = proxy_response.content
                    image = Image.open(BytesIO(content))

                    # Preparar la imagen para subirla
                    img_bytes = BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)

                    # PocketBase usa el campo "imagen" según tu esquema
                    files = {
                        'imagen': ('generated_image.jpg', img_bytes, 'image/jpeg')
                    }

                    # Enviar a PocketBase
                    upload_url = f"{POCKETBASE_URL}/api/collections/{POCKETBASE_COLLECTION}/records"
                    upload_response = await client.post(upload_url, files=files)
                    
                    if upload_response.status_code != 200:
                        raise HTTPException(status_code=upload_response.status_code, detail=f"PocketBase error: {upload_response.text}")

                    res_json = upload_response.json()
                    record_id = res_json['id']
                    filename = res_json['imagen']
                    collection_id = res_json['collectionId']

                    # Construir la URL pública de la imagen en PocketBase
                    pb_image_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
                    
                    image_urls.append({
                        "id": record_id,
                        "url": pb_image_url
                    })
                else:
                    raise HTTPException(status_code=proxy_response.status_code, detail="Error fetching image from OpenAI")

        return {"images": image_urls}
    
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
