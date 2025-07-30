from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Importar CORSMiddleware
from app.routes import router

app = FastAPI(title="Fútbol Predictor", version="1.0.0")

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)