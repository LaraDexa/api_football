from app.ml.pass_predictor import (
    predecir_estadisticas_para_todos,
    predecir_estadisticas_por_jugador
    )
from app.ml.shoot_predictor import (
    predecir_estadisticas_ofensivas
    )

def predecir_estadisticas_para_todos_pase():
    return predecir_estadisticas_para_todos()

def predecir_todas_estadisticas_ofensivas():
    return predecir_estadisticas_ofensivas()

def predecir_estradistica_por_jugador(player_id: int):
    return predecir_estadisticas_por_jugador(player_id)

