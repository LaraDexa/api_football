from fastapi import APIRouter
from app.models.schemas import PlayerRequest
from app.services.predictor_service import predecir_estadisticas_para_todos_pase, predecir_estradistica_por_jugador
from app.services.predictor_service import predecir_todas_estadisticas_ofensivas

router = APIRouter()



@router.get("/predict/pass/all_players")
def predecir_todas_las_estadisticas_todos():
    return predecir_estadisticas_para_todos_pase()

@router.get("/predict/ofensive/all_players")
def predecir_ofensivas():
    return predecir_todas_estadisticas_ofensivas()

@router.get("/predict/pass/{player_id}")
def predecir_pass_por_player(player_id: int):
    return predecir_estradistica_por_jugador(player_id)

