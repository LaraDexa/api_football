
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Fútbol Predictor", version="1.0.0")

app.include_router(router)
