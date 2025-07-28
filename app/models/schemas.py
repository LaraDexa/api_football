from pydantic import BaseModel

class PlayerRequest(BaseModel):
    player_id: int

class PassPredictionRequest(BaseModel):
    player_id: int
    target: str
