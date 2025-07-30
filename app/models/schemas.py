from pydantic import BaseModel
from typing import List


class PlayerRequest(BaseModel):
    player_id: int

class PassPredictionRequest(BaseModel):
    player_id: int
    target: str

class ModelRequest(BaseModel):
    player_id: int
    models: List[str]