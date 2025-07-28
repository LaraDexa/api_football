# app/main.py

from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Fútbol Predictor", version="1.0.0")

# Ya no necesitamos hooks de startup ni imports de train_global_* que ya no existen
app.include_router(router)
