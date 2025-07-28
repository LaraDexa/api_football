from app.ml.pass_predictor import (
    predecir_estadisticas_para_todos
    )
from app.ml.shoot_predictor import (
    predecir_estadisticas_ofensivas
    )

def predecir_estadisticas_para_todos_pase():
    return predecir_estadisticas_para_todos()

def predecir_todas_estadisticas_ofensivas():
    return predecir_estadisticas_ofensivas()
