# app/ml/shoot_predictor.py

import pandas as pd
from pathlib import Path
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor

DATA_PATH = Path("data/DATASET_BARCELONA_PROYECT.xlsx")

TARGETS_SHOOT = ["Sh","SoT","Gls","xG"]
FEATURES = [
    "Min","Touches","Carries","Venue_enc","Weekday","xAG","SCA",
    "Touch_per_Min","Carries_per_Touch","SCA_per_Touch",
    "Rolling_Avg_xAG","Avg_Min_Last_3","Games_Played",
    "Carries_per_90","SCA_per_90","Min_per_Game","xAG_diff",
    "Rolling_CmpPct","delta_Min"
]

BASE_MODELS = {
    "Sh":  RandomForestRegressor(random_state=42),
    "SoT": RandomForestRegressor(random_state=42),
    "Gls": LinearRegression(),
    "xG":  LinearRegression()
}

_pipeline_cache = {}

def _load_and_prepare():
    df = pd.read_excel(DATA_PATH).dropna(subset=TARGETS_SHOOT)
    df["Date"]    = pd.to_datetime(df["Date"])
    df["jornada"] = df["Round"].str.extract(r"(\d+)").astype(int)
    df.sort_values(["jornada","Name","Date"], inplace=True)

    # misma ingeniería que en pass_predictor...
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

    df[FEATURES] = df.groupby("Id")[FEATURES].transform(lambda g: g.fillna(g.mean()))
    df[FEATURES] = df[FEATURES].fillna(0)
    return df

def _train_global(jornada: int):
    if jornada in _pipeline_cache:
        return _pipeline_cache[jornada]

    df = _load_and_prepare()
    train_df = df[df["jornada"] <= jornada] or df

    pipelines = {}
    for t, model in BASE_MODELS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("est",    model)
        ])
        pipe.fit(train_df[FEATURES], train_df[t])
        pipelines[t] = pipe

    _pipeline_cache[jornada] = (pipelines, df)
    return pipelines, df

def _train_player(df_p: pd.DataFrame):
    pipelines = {}
    for t, model in BASE_MODELS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("est",    model)
        ])
        pipe.fit(df_p[FEATURES], df_p[t])
        pipelines[t] = pipe
    return pipelines

def predict_shoot(player_id: int, jornada: int) -> dict:
    df = _load_and_prepare()
    df_p = df[(df["Id"]==player_id)&(df["jornada"]<=jornada-1)]
    if len(df_p) >= 5:
        pipelines = _train_player(df_p)
        df_test   = df
    else:
        pipelines, df_test = _train_global(max(1,jornada-1))

    mask = (df_test["Id"]==player_id)&(df_test["jornada"]==jornada)
    if mask.any():
        row, fb = df_test.loc[mask], False
    else:
        sub = df_test[(df_test["Id"]==player_id)&(df_test["jornada"]<jornada)]
        if sub.empty:
            return {}
        last_j = sub["jornada"].max()
        row    = sub[sub["jornada"]==last_j]
        fb     = True

    out = {}
    for t, pipe in pipelines.items():
        pred = pipe.predict(row[FEATURES])[0]
        if not fb:
            real = float(row[t].iloc[0])
            acc  = round((1-abs(pred-real)/real)*100,2) if real else None
        else:
            real = None; acc = None
        out[t] = {
            "predicted":  round(pred,2),
            "real":       real,
            "accuracy":   acc,
            "model_used": pipe.named_steps["est"].__class__.__name__
        }
    return out

def append_stats_and_retrain(new_row: dict, jornada: int) -> bool:
    df0    = pd.read_excel(DATA_PATH)
    new_df = pd.DataFrame([new_row])
    df1    = pd.concat([df0, new_df], ignore_index=True)
    df1.to_excel(DATA_PATH, index=False)
    _pipeline_cache.clear()
    return True

def train_global_shoot(jornada: int):
    return _train_global(jornada)
