import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut

# Carga del dataset
df = pd.read_excel("data/DATASET_BARCELONA_PROYECT.xlsx")
df = df.dropna(subset=["Cmp", "Att", "Cmp%", "PrgP"])

# Hiperparámetros
targets = ["Cmp", "Att", "Cmp%", "PrgP"]
RF_PARAMS    = {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2]}
RIDGE_PARAMS = {"alpha": [0.1, 1.0, 10.0, 100.0]}
KNN_PARAMS   = {"n_neighbors": [2, 5, 10], "weights": ["uniform", "distance"]}

# Función de tuning de hiperparámetros
def tune_model(base_model, param_grid, X, y):
    """GridSearch sobre MAE, sin paralelismo (-n_jobs=1)."""
    if isinstance(base_model, KNeighborsRegressor):
        max_k   = len(y)
        knn_vals= [k for k in param_grid["n_neighbors"] if k <= max_k]
        param_grid = {
            "n_neighbors": knn_vals or [max_k],
            "weights":     param_grid.get("weights", ["uniform"]),
        }
    gs = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,    # <- un solo hilo
    )
    gs.fit(X, y)
    return gs.best_estimator_

# Selección de modelo según número de partidos
def elegir_modelo(n_partidos: int):
    if n_partidos >= 10:
        return RandomForestRegressor(random_state=42)
    elif n_partidos >= 8:
        return Ridge()
    else:
        return BayesianRidge()

# Ingeniería de features
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Name", "Date"])

df["Touch_per_Min"]       = df["Touches"] / df["Min"].replace(0, 1)
df["Carries_per_Touch"]   = df["Carries"] / df["Touches"].replace(0, 1)
df["SCA_per_Touch"]       = df["SCA"] / df["Touches"].replace(0, 1)
df["delta_Min"]           = df.groupby("Name")["Min"].diff().fillna(0)

rolling = df.groupby("Name")
df["Rolling_Avg_xAG"]     = rolling["xAG"].rolling(3, 1).mean().reset_index(level=0, drop=True)
df["Avg_Min_Last_3"]      = rolling["Min"].rolling(3, 1).mean().reset_index(level=0, drop=True)

df["Games_Played"]         = rolling.cumcount() + 1
df["Carries_per_90"]      = df["Carries"] / (df["Min"].replace(0, 1) / 90)
df["SCA_per_90"]          = df["SCA"] / (df["Min"].replace(0, 1) / 90)
df["Min_per_Game"]        = rolling["Min"].transform("mean")
df["xAG_diff"]            = df["xAG"] - df["Ast"]
df["Rolling_CmpPct"]      = rolling["Cmp%"].rolling(3, 1).mean().reset_index(level=0, drop=True)

df["Weekday"] = df["Date"].dt.weekday
le = LabelEncoder()
df["Venue_enc"] = le.fit_transform(df["Venue"])

features = [
    "Min", "Touches", "Carries", "Venue_enc", "Weekday", "xAG", "SCA",
    "Touch_per_Min", "Carries_per_Touch", "SCA_per_Touch",
    "Rolling_Avg_xAG", "Avg_Min_Last_3", "Games_Played",
    "Carries_per_90", "SCA_per_90", "Min_per_Game", "xAG_diff", "Rolling_CmpPct", "delta_Min"
]

# Escalado de features
df_scaled = df.copy()
scaler = StandardScaler()
df_scaled[features] = scaler.fit_transform(df[features])

# 4. Entrenamiento global de fallback
global_models = {}

def train_global_pass_models():
    for t in targets:
        X_all = df_scaled[features]
        y_all = df_scaled[t]
        low, up = y_all.quantile(0.10), y_all.quantile(0.90)
        y_clip  = y_all.clip(low, up)
        best_rf = tune_model(
            RandomForestRegressor(random_state=42),
            RF_PARAMS,
            X_all, y_clip
        )
        global_models[t] = best_rf
        
train_global_pass_models()

# Función de predicción para todos los jugadores
def predecir_estadisticas_para_todos(predict_match: int = 21):
    results = {"players": []}

    for pid in df_scaled["Id"].unique():
        dfp = df_scaled[df_scaled["Id"] == pid]\
                 .sort_values("Date")\
                 .reset_index(drop=True)
        if len(dfp) < 3:
            continue

        m     = min(len(dfp), predict_match)
        train = dfp.iloc[:m-1]
        test  = dfp.iloc[m-1 : m]

        warns = []
        if len(train) < 10:
            warns.append("menos de 10 partidos")

        preds = {}
        for t in targets:
            X_train, y_train = train[features], train[t]
            low, up = y_train.quantile(0.10), y_train.quantile(0.90)
            y_train = y_train.clip(low, up)

            X_test  = test[features].values.reshape(1, -1)
            y_true  = test[t].iloc[0]

            base = elegir_modelo(len(train))
            if isinstance(base, RandomForestRegressor):
                params = RF_PARAMS
            elif isinstance(base, Ridge):
                params = RIDGE_PARAMS
            else:
                params = KNN_PARAMS

            model = tune_model(base, params, X_train, y_train)

            cv     = LeaveOneOut() if len(y_train) < 5 else 3
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,     # <- un sólo hilo
            )
            mae_cv = -np.nanmean(scores)

            pred     = model.predict(X_test)[0]
            y_pred_tr= model.predict(X_train)
            r2       = r2_score(y_train, y_pred_tr)
            mae_tr   = mean_absolute_error(y_train, y_pred_tr)

            # fallback si cv peor que global
            X_all    = df_scaled[features]
            mae_gl   = mean_absolute_error(
                df_scaled[t],
                global_models[t].predict(X_all)
            )
            if np.isnan(mae_cv) or mae_cv > mae_gl:
                model     = global_models[t]
                pred      = model.predict(X_test)[0]
                y_pred_tr = model.predict(X_train)
                r2        = r2_score(y_train, y_pred_tr)
                mae_tr    = mean_absolute_error(y_train, y_pred_tr)

            # importance
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = abs(model.coef_)
            else:
                imps = [None]*len(features)

            feats_imp = sorted(
                [{"feature":f, "importance":None if v is None else round(v,4)}
                 for f,v in zip(features, imps)],
                key=lambda x: x["importance"] or 0, reverse=True
            )

            preds[t] = {
                "predicted_value": round(float(pred),2),
                "real_value":       float(y_true),
                "r2_train":         round(r2,3),
                "mae_train":        round(mae_tr,2),
                "mae_cv":           round(mae_cv,2),
                "model_used":       model.__class__.__name__,
                "feature_importance": feats_imp
            }

        results["players"].append({
            "player_id":   int(pid),
            "player_name": dfp["Name"].iloc[0],
            "match_number": m,
            "games_used":  len(train),
            "warnings":    warns,
            "predictions": preds
        })

    return results

# Nueva función: predicción para un solo jugador por su ID
def predecir_estadisticas_por_jugador(player_id: int, predict_match: int = 21):
    sub = df_scaled[df_scaled["Id"] == player_id].sort_values("Date").reset_index(drop=True)
    if len(sub) < 3:
        return {"player_id": player_id, "player_name": None, "predictions": {}}
    m = min(len(sub), predict_match)
    test = sub.iloc[m-1:m]
    name = sub["Name"].iloc[0]
    preds = {}
    for t in targets:
        y_true = test[t].iloc[0]
        model = global_models[t]
        pred = model.predict(test[features].values.reshape(1, -1))[0]
        preds[t] = {"real_value": float(y_true), "predicted_value": round(float(pred),2)}
    return {"player_id": player_id, "player_name": name, "predictions": preds}