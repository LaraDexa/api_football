# app/services/predictor_service.py

from typing import Any, Dict

from app.ml.pass_predictor     import predict_player as predict_pass, append_stats_and_retrain as append_pass
from app.ml.shoot_predictor    import predict_shoot
from app.ml.dribble_predictor  import predict_dribble
from app.ml.defense_predictor  import predict_defense

def get_all_stats(player_id: int, jornada: int) -> Dict[str, Any]:
    base = predict_pass(player_id, jornada)
    if not base:
        return {}

    base["stats"]["tiro"]    = predict_shoot(player_id, jornada)   or {}
    base["stats"]["regate"]  = predict_dribble(player_id, jornada) or {}
    defensa                  = predict_defense(player_id, jornada) or {}
    base["stats"]["defensa"] = defensa.get("defensa", {})
    if "note_def" in defensa:
        base["stats"]["note_def"] = defensa["note_def"]

    return base

def add_real_stats(player_id: int, stats: Dict[str, Any]) -> bool:
    m = stats["pase"]["match_number"]
    ok1 = append_pass({
        "Id": player_id,
        "Cmp": stats["pase"]["Cmp"],
        "Att": stats["pase"]["Att"],
        "Cmp%": stats["pase"]["Cmp_percent"],
        "PrgP": stats["pase"]["PrgP"],
        "Round": f"Matchweek {m}"
    }, m)
    return ok1
