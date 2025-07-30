# app/services/predictor_service.py

from typing import Any, Dict
from app.ml.pass_predictor     import predict_player as predict_pass
from app.ml.shoot_predictor    import predict_shoot
from app.ml.dribble_predictor  import predict_dribble
from app.ml.pass_predictor     import append_stats_and_retrain as append_pass
# from app.ml.shoot_predictor    import append_stats_and_retrain as append_shoot
# from app.ml.dribble_predictor  import append_stats_and_retrain as append_dribble

def get_all_stats(player_id: int, jornada: int) -> Dict[str, Any]:
    base = predict_pass(player_id, jornada)
    if not base:
        return {}
    # añade tiro y regate
    tiros   = predict_shoot(player_id, jornada)
    regates = predict_dribble(player_id, jornada)
    base["stats"]["tiro"]   = tiros
    base["stats"]["regate"] = regates
    return base

def add_real_stats(player_id: int, stats: Dict[str, Any]) -> bool:
    # stats debe contener {pass:…, shoot:…, dribble:…} con match_number
    m = stats["pase"]["match_number"]
    ok1 = append_pass({
        "Id": player_id,
        "Cmp": stats["pase"]["Cmp"],
        "Att": stats["pase"]["Att"],
        "Cmp%": stats["pase"]["Cmp_percent"],
        "PrgP": stats["pase"]["PrgP"],
        "Round": f"Matchweek {m}"
    }, m)
    # ok2 = append_shoot({
    #     "Id": player_id,
    #     **{t: stats["tiro"][t] for t in stats["tiro"]},
    #     "Round": f"Matchweek {m}"
    # }, m)
    # ok3 = append_dribble({
    #     "Id": player_id,
    #     **{t: stats["regate"][t] for t in stats["regate"]},
    #     "Round": f"Matchweek {m}"
    # }, m)
    return ok1 
# and ok2 and ok3
