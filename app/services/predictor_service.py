# app/services/predictor_service.py

from typing import Any, Dict

from app.ml.pass_predictor     import predict_player as predict_pass, append_stats_and_retrain as append_pass
from app.ml.shoot_predictor    import predict_shoot, append_stats_and_retrain as append_shoot
from app.ml.dribble_predictor  import predict_dribble, append_stats_and_retrain as append_dribble
from app.ml.defense_predictor  import predict_defense, append_stats_and_retrain as append_defense

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

    # 1) Pases
    pass_row = {
        "Id":    player_id,
        "Cmp":   stats["pase"]["Cmp"],
        "Att":   stats["pase"]["Att"],
        "Cmp%":  stats["pase"]["Cmp_percent"],
        "PrgP":  stats["pase"]["PrgP"],
        "Round": f"Matchweek {m}"
    }
    ok1 = append_pass(pass_row, m)

    # 2) Tiro
    shoot_row = {
        "Id":    player_id,
        "Sh":    stats["tiro"]["Sh"],
        "SoT":   stats["tiro"]["SoT"],
        "Gls":   stats["tiro"]["Gls"],
        "xG":    stats["tiro"]["xG"],
        "Round": f"Matchweek {m}"
    }
    ok2 = append_shoot(shoot_row, m)

    # 3) Regate
    dribble_row = {
        "Id":      player_id,
        "Carries": stats["regate"]["Carries"],
        "PrgC":    stats["regate"]["PrgC"],
        "Att.1":   stats["regate"]["Att.1"],
        "Succ":    stats["regate"]["Succ"],
        "Round":   f"Matchweek {m}"
    }
    ok3 = append_dribble(dribble_row, m)

    # 4) Defensa
    defense_row = {
        "Id":     player_id,
        "Tkl":    stats["defensa"]["Tkl"],
        "Int":    stats["defensa"]["Int"],
        "Blocks": stats["defensa"]["Blocks"],
        "Round":  f"Matchweek {m}"
    }
    ok4 = append_defense(defense_row, m)

    return ok1 and ok2 and ok3 and ok4

