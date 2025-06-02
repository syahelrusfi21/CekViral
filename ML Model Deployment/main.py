# cekviral_project/main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys

# Tambahkan direktori root proyek ke sys.path secara eksplisit
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

# Import the main router from your application
from app.api.endpoints import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="CekViral: Asisten Cerdas untuk Verifikasi Konten Viral"
)

# Include API endpoints
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Selamat datang di CekViral API! Kunjungi /docs untuk dokumentasi API."}

# Event handler untuk loading model ML di sini
@app.on_event("startup")
async def startup_event():
    print("Aplikasi CekViral startup. Memuat model ML...")
    from app.services.ml_model import load_ml_model
    load_ml_model() # Muat model saat startup
    print("Model ML berhasil dimuat.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Aplikasi CekViral shutdown.")

if __name__ == "__main__":
    import uvicorn
    # Jalankan dengan: uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)