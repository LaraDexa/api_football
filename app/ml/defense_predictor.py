# app/ml/defense_predictor.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, LeaveOneOut
from sklearn.ensemble        import RandomForestRegressor
from sklearn.neighbors       import KNeighborsRegressor

DATA_PATH = Path("data/DATASET_BARCELONA_PROYECT.xlsx")

TARGETS_DEF = ["Tkl", "Int", "Blocks"]
FEATURES    = [
    "Min","Touches","Carries","Venue_enc","Weekday","xAG","SCA",
    "Touch_per_Min","Carries_per_Touch","SCA_per_Touch",
    "Rolling_Avg_xAG","Avg_Min_Last_3","Games_Played",
    "Carries_per_90","SCA_per_90","Min_per_Game","xAG_diff",
    "Rolling_CmpPct","delta_Min"
]

BASE_MODELS = {
    "Tkl":    RandomForestRegressor(random_state=42),
    "Int":    RandomForestRegressor(random_state=42),
    "Blocks": KNeighborsRegressor()
}
PARAM_GRIDS = {
    "Tkl": {
        "est__n_estimators":     [100,200],
        "est__max_depth":        [5,10],
        "est__min_samples_split":[2,5],
        "est__max_features":     ["sqrt","log2"],
        "est__bootstrap":        [True,False]
    },
    "Int": {
        "est__n_estimators":     [100,200],
        "est__max_depth":        [5,10],
        "est__min_samples_split":[2,5],
        "est__max_features":     ["sqrt","log2"],
        "est__bootstrap":        [True,False]
    },
    "Blocks": {
        "est__n_neighbors": [1,2,3,5,7,10],
        "est__weights":     ["uniform","distance"]
    }
}

_pipeline_cache_def = {}

def _load_and_prepare_def():
    df = pd.read_excel(DATA_PATH).dropna(subset=TARGETS_DEF)
    df["Date"]    = pd.to_datetime(df["Date"])
    df["jornada"] = df["Round"].str.extract(r"(\d+)").astype(int)
    df.sort_values(["jornada","Name","Date"], inplace=True)

    # misma ingeniería
    df["Weekday"]           = df["Date"].dt.weekday
    df["Touch_per_Min"]     = df["Touches"].div(df["Min"].replace(0,1))
    df["Carries_per_Touch"] = df["Carries"].div(df["Touches"].replace(0,1))
    df["SCA_per_Touch"]     = df["SCA"].div(df["Touches"].replace(0,1))
    df["delta_Min"]         = df.groupby("Name")["Min"].diff().fillna(0)
    rolling = df.groupby("Name")
    df["Rolling_Avg_xAG"]   = rolling["xAG"].rolling(3,1).mean().reset_index(level=0,drop=True)
    df["Avg_Min_Last_3"]    = rolling["Min"].rolling(3,1).mean().reset_index(level=0,drop=True)
    df["Games_Played"]      = rolling.cumcount()+1
    df["Carries_per_90"]    = df["Carries"].div(df["Min"].replace(0,1)/90)
    df["SCA_per_90"]        = df["SCA"].div(df["Min"].replace(0,1)/90)
    df["Min_per_Game"]      = rolling["Min"].transform("mean")
    df["xAG_diff"]          = df["xAG"] - df["Ast"]
    df["Rolling_CmpPct"]    = rolling["Cmp%"].rolling(3,1).mean().reset_index(level=0,drop=True)

    # Venue encoding
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[["Venue_enc"]] = enc.fit_transform(df[["Venue"]])

    # 1) imputar por media del mismo jugador
    df[FEATURES] = df.groupby("Id")[FEATURES].transform(lambda g: g.fillna(g.mean()))
    # 2) rellenar restantes con 0
    df[FEATURES] = df[FEATURES].fillna(0)

    return df

def _train_global_def(jornada: int):
    if jornada in _pipeline_cache_def:
        return _pipeline_cache_def[jornada]
    df       = _load_and_prepare_def()
    train_df = df[df["jornada"] <= jornada]
    if train_df.empty:
        train_df = df.copy()

    cv        = TimeSeriesSplit(n_splits=5)
    pipelines = {}
    for t in TARGETS_DEF:
        pipe = Pipeline([("scaler", StandardScaler()), ("est", BASE_MODELS[t])])
        gs   = GridSearchCV(pipe, PARAM_GRIDS[t], cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(train_df[FEATURES], train_df[t])
        pipelines[t] = gs.best_estimator_

    _pipeline_cache_def[jornada] = (pipelines, df)
    return pipelines, df

def _train_player_def(df_p: pd.DataFrame):
    n   = len(df_p)
    cv  = LeaveOneOut() if n < 5 else TimeSeriesSplit(n_splits=3)
    pipes = {}
    for t in TARGETS_DEF:
        pipe = Pipeline([("scaler", StandardScaler()), ("est", BASE_MODELS[t])])
        gs   = GridSearchCV(pipe, PARAM_GRIDS[t], cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(df_p[FEATURES], df_p[t])
        pipes[t] = gs.best_estimator_
    return pipes

def predict_defense(player_id: int, jornada: int) -> dict:
    df   = _load_and_prepare_def()
    df_p = df[(df["Id"]==player_id)&(df["jornada"] <= jornada-1)]
    if len(df_p) >= 5:
        pipelines = _train_player_def(df_p)
    else:
        pipelines, df = _train_global_def(max(1, jornada-1))

    mask = (df["Id"]==player_id)&(df["jornada"]==jornada)
    if mask.any():
        row, fb = df.loc[mask], False
    else:
        sub        = df[df["Id"]==player_id]
        sub        = sub[sub["jornada"]<jornada]
        if sub.empty:
            return {}
        last_j     = sub["jornada"].max()
        row, fb    = sub[sub["jornada"]==last_j], True

    out = {}
    for t, pipe in pipelines.items():
        pred = pipe.predict(row[FEATURES])[0]
        real = float(row[t].iloc[0]) if not fb else None
        acc  = round((1-abs(pred-real)/real)*100,2) if real else None
        out[t] = {
            "predicted":  round(pred,2),
            "real":       real,
            "accuracy":   acc,
            "model_used": pipe.named_steps["est"].__class__.__name__
        }

    return {
        "player_id": player_id,
        "jornada":   jornada,
        "defensa":   out,
        **({"note_def": f"No jugó jornada {jornada}, usando última jornada {last_j}"} if fb else {})
    }

def append_stats_and_retrain(new_row: dict, jornada: int) -> bool:
    df0    = pd.read_excel(DATA_PATH)
    new_df = pd.DataFrame([new_row])
    df1    = pd.concat([df0, new_df], ignore_index=True)
    df1.to_excel(DATA_PATH, index=False)
    _pipeline_cache_def.clear()
    return True

def train_global_def(jornada: int):
    return _train_global_def(jornada)
