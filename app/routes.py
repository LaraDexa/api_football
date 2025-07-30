# app/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from app.services.predictor_service import get_all_stats, add_real_stats

router = APIRouter()

@router.get("/predict/player/{player_id}/{jornada}")
async def predict_player(player_id: int, jornada: int) -> Dict[str, Any]:
    res = get_all_stats(player_id, jornada)
    if not res:
        raise HTTPException(404, "No hay datos para ese jugador/jornada")
    return res

class RealStats(BaseModel):
    match_number: int
    # pase
    Cmp: float; Att: float; Cmp_percent: float; PrgP: float
    # tiro
    Sh:   float; SoT: float; Gls: float; xG: float
    # regate
    Carries: float; PrgC: float; Att_1: float; Succ: float

@router.post("/predict/player/{player_id}/real")
async def submit_real_all(player_id: int, s: RealStats) -> Dict[str, Any]:
    # adaptamos nombres para el diccionario
    stats = {
        "pase": {
            "match_number": s.match_number,
            "Cmp": s.Cmp, "Att": s.Att,
            "Cmp_percent": s.Cmp_percent, "PrgP": s.PrgP
        },
        "tiro": {
            "match_number": s.match_number,
            "Sh": s.Sh, "SoT": s.SoT,
            "Gls": s.Gls, "xG": s.xG
        },
        "regate": {
            "match_number": s.match_number,
            "Carries": s.Carries, "PrgC": s.PrgC,
            "Att.1": s.Att_1, "Succ": s.Succ
        }
    }
    ok = add_real_stats(player_id, stats)
    if not ok:
        raise HTTPException(500, "Error al guardar estadísticas reales")
    return {"ok": True, "message": "Estadísticas reales guardadas para pase, tiro y regate"}
