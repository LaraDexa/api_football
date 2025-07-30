# app/ml/pass_predictor.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, LeaveOneOut
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.neighbors        import KNeighborsRegressor

DATA_PATH = Path("data/DATASET_BARCELONA_PROYECT.xlsx")

TARGETS = ["Cmp", "Att", "Cmp%", "PrgP"]
FEATURES = [
    "Min","Touches","Carries","Venue_enc","Weekday","xAG","SCA",
    "Touch_per_Min","Carries_per_Touch","SCA_per_Touch",
    "Rolling_Avg_xAG","Avg_Min_Last_3","Games_Played",
    "Carries_per_90","SCA_per_90","Min_per_Game","xAG_diff",
    "Rolling_CmpPct","delta_Min"
]

BASE_MODELS = {
    "Cmp":  RandomForestRegressor(random_state=42),
    "Att":  RandomForestRegressor(random_state=42),
    "Cmp%": LinearRegression(),
    "PrgP": KNeighborsRegressor()
}
PARAM_GRIDS = {
    "Cmp": {
        "est__n_estimators":     [100,200],
        "est__max_depth":        [5,10],
        "est__min_samples_split":[2,5],
        "est__max_features":     ["sqrt","log2"],
        "est__bootstrap":        [True,False]
    },
    "Att": {
        "est__n_estimators":     [50,100],
        "est__max_depth":        [None,5],
        "est__min_samples_split":[2,5],
        "est__max_features":     ["sqrt","log2"],
        "est__bootstrap":        [True,False]
    },
    "Cmp%": {
        "est__fit_intercept": [True,False]
    },
    "PrgP": {
        "est__n_neighbors": [1,2,3,5,7,10],
        "est__weights":     ["uniform","distance"]
    }
}

_pipeline_cache = {}

def _load_and_prepare():
    df = pd.read_excel(DATA_PATH).dropna(subset=TARGETS)
    df["Date"]    = pd.to_datetime(df["Date"])
    df["jornada"] = df["Round"].str.extract(r"(\d+)").astype(int)
    df.sort_values(["jornada","Name","Date"], inplace=True)

    # Ingeniería de features
    df["Weekday"]           = df["Date"].dt.weekday
    df["Touch_per_Min"]     = df["Touches"].div(df["Min"].replace(0,1))
    df["Carries_per_Touch"] = df["Carries"].div(df["Touches"].replace(0,1))
    df["SCA_per_Touch"]     = df["SCA"].div(df["Touches"].replace(0,1))
    df["delta_Min"]         = df.groupby("Name")["Min"].diff().fillna(0)
    rolling = df.groupby("Name")
    df["Rolling_Avg_xAG"]   = rolling["xAG"].rolling(3,1).mean().reset_index(level=0,drop=True)
    df["Avg_Min_Last_3"]    = rolling["Min"].rolling(3,1).mean().reset_index(level=0,drop=True)
    df["Games_Played"]      = rolling.cumcount() + 1
    df["Carries_per_90"]    = df["Carries"].div(df["Min"].replace(0,1)/90)
    df["SCA_per_90"]        = df["SCA"].div(df["Min"].replace(0,1)/90)
    df["Min_per_Game"]      = rolling["Min"].transform("mean")
    df["xAG_diff"]          = df["xAG"] - df["Ast"]
    df["Rolling_CmpPct"]    = rolling["Cmp%"].rolling(3,1).mean().reset_index(level=0,drop=True)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[["Venue_enc"]] = enc.fit_transform(df[["Venue"]])

    # imputación por jugador y relleno de NaN
    df[FEATURES] = df.groupby("Id")[FEATURES].transform(lambda g: g.fillna(g.mean()))
    df[FEATURES] = df[FEATURES].fillna(0)

    return df

def _train_global(jornada: int):
    if jornada in _pipeline_cache:
        return _pipeline_cache[jornada]

    df = _load_and_prepare()
    train_df = df[df["jornada"] <= jornada]
    if train_df.empty:
        train_df = df.copy()

    tscv = TimeSeriesSplit(n_splits=5)
    pipelines = {}
    for t in TARGETS:
        pipe = Pipeline([("scaler", StandardScaler()), ("est", BASE_MODELS[t])])
        gs   = GridSearchCV(pipe, PARAM_GRIDS[t], cv=tscv,
                            scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(train_df[FEATURES], train_df[t])
        pipelines[t] = gs.best_estimator_

    _pipeline_cache[jornada] = (pipelines, df)
    return pipelines, df

def _train_player(df_player: pd.DataFrame):
    n = len(df_player)
    cv = LeaveOneOut() if n < 5 else TimeSeriesSplit(n_splits=3)
    pipelines = {}
    for t in TARGETS:
        pipe = Pipeline([("scaler", StandardScaler()), ("est", BASE_MODELS[t])])
        gs   = GridSearchCV(pipe, PARAM_GRIDS[t], cv=cv,
                            scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(df_player[FEATURES], df_player[t])
        pipelines[t] = gs.best_estimator_
    return pipelines

def predict_player(player_id: int, jornada: int) -> dict:
    df = _load_and_prepare()

    df_player = df[(df["Id"] == player_id) & (df["jornada"] <= jornada - 1)]
    if len(df_player) >= 5:
        pipelines   = _train_player(df_player)
        df_for_test = df
    else:
        pipelines, df_for_test = _train_global(max(1,jornada - 1))

    mask_exact = (df_for_test["Id"] == player_id) & (df_for_test["jornada"] == jornada)
    if mask_exact.any():
        test_row    = df_for_test.loc[mask_exact]
        is_fallback = False
    else:
        sub         = df_for_test[df_for_test["Id"] == player_id]
        sub         = sub[sub["jornada"] < jornada]
        if sub.empty:
            return {}
        last_j      = sub["jornada"].max()
        test_row    = sub[sub["jornada"] == last_j]
        is_fallback = True

    name   = test_row["Name"].iloc[0]
    X_test = test_row[FEATURES]
    out    = {}
    for t, pipe in pipelines.items():
        pred = pipe.predict(X_test)[0]
        if is_fallback:
            real, acc = None, None
        else:
            real = float(test_row[t].iloc[0])
            acc  = round((1 - abs(pred - real) / real) * 100, 2) if real else None
        out[t] = {
            "predicted": round(pred,2),
            "real":      real,
            "accuracy": acc,
            "model_used": pipe.named_steps["est"].__class__.__name__
        }

    res = {
        "player_id":   player_id,
        "player_name": name,
        "jornada":     jornada,
        "stats":       {"pase": out}
    }
    if is_fallback:
        res["note"] = f"No jugó jornada {jornada}, usando última jornada {last_j}"
    return res

def append_stats_and_retrain(new_row: dict, jornada: int) -> bool:
    df0    = pd.read_excel(DATA_PATH)
    new_df = pd.DataFrame([new_row])
    df1    = pd.concat([df0, new_df], ignore_index=True)
    df1.to_excel(DATA_PATH, index=False)
    _pipeline_cache.clear()
    return True

def train_global(jornada: int):
    return _train_global(jornada)
