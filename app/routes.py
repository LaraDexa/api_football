from fastapi import APIRouter
# from app.models.schemas import PlayerRequest, PassPredictionRequest
from app.services.predictor_service import predecir_estadisticas_para_todos_pase
from app.services.predictor_service import predecir_todas_estadisticas_ofensivas

router = APIRouter()

# @router.post("/predict/pass/all")
# def predecir_todas_las_estadisticas(request: PlayerRequest):
#     return predecir_todas_las_estadisticas_de_pase(request.player_id)

@router.get("/predict/pass/all_players")
def predecir_todas_las_estadisticas_todos():
    return predecir_estadisticas_para_todos_pase()

@router.get("/predict/ofensive/all_players")
def predecir_ofensivas():
    return predecir_todas_estadisticas_ofensivas()