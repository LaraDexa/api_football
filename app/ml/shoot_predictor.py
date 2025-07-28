# app/ml/shoot_predictor.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import GridSearchCV

# 1. Leo datos
DATASET_PATH = "data/DATASET_BARCELONA_PROYECT.xlsx"
df = pd.read_excel(DATASET_PATH)

# 2. Defino variables objetivo
OFF_TARGETS = ["Sh", "SoT", "xG", "npxG", "Gls"]
df = df.dropna(subset=OFF_TARGETS)

# 3. Ingeniería de features
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Name", "Date"])
df["Touch_per_Min"]     = df["Touches"] / df["Min"].replace(0, 1)
df["Carries_per_Touch"] = df["Carries"] / df["Touches"].replace(0, 1)
df["SCA_per_Touch"]     = df["SCA"]    / df["Touches"].replace(0, 1)
df["delta_Min"]         = df.groupby("Name")["Min"].diff().fillna(0)

rolling = df.groupby("Name")
for col in ["Sh", "xG", "SoT"]:
    df[f"Rolling_{col}_Last_3"] = (
        rolling[col]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

df["Games_Played"]   = rolling.cumcount() + 1
df["Carries_per_90"] = df["Carries"] / (df["Min"].replace(0, 1) / 90)
df["SCA_per_90"]     = df["SCA"]     / (df["Min"].replace(0, 1) / 90)
df["Min_per_Game"]   = rolling["Min"].transform("mean")
df["xG_diff"]        = df["xG"] - df["Gls"]
df["Rolling_xG_Pct"] = (
    rolling["xG"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df["Weekday"] = df["Date"].dt.weekday
df["Start"]   = df["Start"].map({"Y": 1, "N": 0}).fillna(0)

# POS one-hot con handle_unknown para CV
ohe_pos  = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
pos_ohe  = ohe_pos.fit_transform(df[["Pos"]])
pos_cols = [f"Pos_{c}" for c in ohe_pos.categories_[0][1:]]
df[pos_cols] = pos_ohe

# 4. Lista de features
NUM_FEATS = [
    "Min","Touches","Carries","xAG","SCA",
    "Touch_per_Min","Carries_per_Touch","SCA_per_Touch","delta_Min",
    "Rolling_Sh_Last_3","Rolling_xG_Last_3","Rolling_SoT_Last_3",
    "Games_Played","Carries_per_90","SCA_per_90","Min_per_Game",
    "xG_diff","Rolling_xG_Pct"
]
CAT_FEATS = ["Venue", "Weekday", "Start"] + pos_cols

# 5. Preprocesado (con handle_unknown)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_FEATS),
])

# 6. Hiperparámetros
RF_PARAMS      = {"regressor__n_estimators": [100, 200], "regressor__max_depth": [None, 5, 10]}
RIDGE_PARAMS   = {"regressor__alpha": [0.1, 1.0, 10.0]}
POISSON_PARAMS = {"regressor__alpha": [1e-2, 1e-1, 1.0]}

# 7. Creamos pipelines y pipelines+GridSearch
models = {}
for t in OFF_TARGETS:
    if t in ["Sh", "SoT", "Gls"]:
        base = Pipeline([
            ("pre", preprocessor),
            ("regressor", PoissonRegressor(max_iter=3000))
        ])
        grid = GridSearchCV(base, POISSON_PARAMS, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
        model = TransformedTargetRegressor(regressor=grid, func=np.log1p, inverse_func=np.expm1)
    else:
        rf_pipe    = Pipeline([("pre", preprocessor), ("regressor", RandomForestRegressor(random_state=42))])
        rf_grid    = GridSearchCV(rf_pipe, RF_PARAMS, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
        model = TransformedTargetRegressor(regressor=rf_grid, func=np.log1p, inverse_func=np.expm1)

    # Entrenamos con TODO el dataset
    model.fit(df, df[t])
    models[t] = model

# 8. Función de predicción (ya con real_value convertido)
def predecir_estadisticas_ofensivas(predict_match: int = 21):
    out = {"players": []}
    for pid in df["Id"].unique():
        sub = df[df["Id"] == pid].sort_values("Date").reset_index(drop=True)
        if len(sub) < 3:
            continue

        m    = min(len(sub), predict_match)
        test = sub.iloc[m - 1 : m]
        preds = {}

        for t in OFF_TARGETS:
            y_true = test[t].iloc[0]
            y_pred = models[t].predict(test)[0]

            preds[t] = {
                # <-- aquí convertimos a Python float
                "real_value":      float(y_true),
                "predicted_value": round(float(y_pred), 2),
            }

        out["players"].append({
            "player_id":   int(pid),
            "player_name": sub["Name"].iloc[0],
            "match_number": int(m),
            "predictions":  preds
        })

    return out
