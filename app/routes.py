# app/routes.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict

from app.services.predictor_service import get_all_stats, add_real_stats

# Public train_global functions
from app.ml.pass_predictor    import train_global      as train_global_pass
from app.ml.shoot_predictor   import train_global_shoot
from app.ml.dribble_predictor import train_global_dribble
from app.ml.defense_predictor import train_global_def

router = APIRouter()


class RealStats(BaseModel):
    match_number: int
    # pase
    Cmp: float; Att: float; Cmp_percent: float; PrgP: float
    # tiro
    Sh:   float; SoT: float; Gls: float; xG: float
    # regate
    Carries: float; PrgC: float; Att_1: float; Succ: float
    # defensa (sin Clr)
    Tkl:   float; Int: float; Blocks: float

@router.post("/predict/player/{player_id}/real")
async def submit_real_all(
    player_id: int,
    s: RealStats,
    bg: BackgroundTasks
) -> Dict[str, Any]:
    stats = {
        "pase": {
            "match_number": s.match_number,
            "Cmp":          s.Cmp,
            "Att":          s.Att,
            "Cmp_percent":  s.Cmp_percent,
            "PrgP":         s.PrgP
        },
        "tiro": {
            "match_number": s.match_number,
            "Sh":   s.Sh,
            "SoT":  s.SoT,
            "Gls":  s.Gls,
            "xG":   s.xG
        },
        "regate": {
            "match_number": s.match_number,
            "Carries": s.Carries,
            "PrgC":    s.PrgC,
            "Att.1":   s.Att_1,
            "Succ":    s.Succ
        },
        "defensa": {
            "match_number": s.match_number,
            "Tkl":    s.Tkl,
            "Int":    s.Int,
            "Blocks": s.Blocks
        }
    }

    ok = add_real_stats(player_id, stats)
    if not ok:
        raise HTTPException(status_code=500, detail="Error al guardar estadísticas reales")

    # Disparamos re‑entrenamiento en segundo plano
    j0 = max(1, s.match_number)
    bg.add_task(train_global_pass,    j0)
    bg.add_task(train_global_shoot,   j0)
    bg.add_task(train_global_dribble, j0)
    bg.add_task(train_global_def,     j0)

    return {"ok": True, "message": "Estadísticas reales guardadas y re‑entrenamiento en segundo plano"}

@router.post("/predict/player/{player_id}/jornada/{jornada}")
async def predict_player(
    player_id: int,
    jornada: int,
    bg: BackgroundTasks
) -> Dict[str, Any]:
    # Encolamos un re‑entreno global hasta jornada-1
    j0 = max(1, jornada - 1)
    bg.add_task(train_global_pass,    j0)
    bg.add_task(train_global_shoot,   j0)
    bg.add_task(train_global_dribble, j0)
    bg.add_task(train_global_def,     j0)

    res = get_all_stats(player_id, jornada)
    if not res:
        raise HTTPException(status_code=404, detail="No hay datos para ese jugador/jornada")
    return res

class ChatRequest(BaseModel):
    prompt: str
    jornada: int

@router.post("/chat")
async def chat_con_usuario(request: ChatRequest):
    from app.services.chatbot_service import procesar_pregunta
    respuesta = await procesar_pregunta(request.prompt, request.jornada)
    return {"respuesta": respuesta}