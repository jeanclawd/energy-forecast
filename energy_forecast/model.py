"""LightGBM + XGBoost ensemble (simple average), one global model per target
with the zone as a feature. A seasonal-naive baseline (same hour last week)
is kept alongside for the analytics."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURES

LGB_PARAMS = dict(n_estimators=800, learning_rate=0.04, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
XGB_PARAMS = dict(n_estimators=700, learning_rate=0.05, max_depth=8, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist")


class Ensemble:
    def __init__(self, target: str):
        self.target = target
        self.features = FEATURES[target]
        self.lgb = lgb.LGBMRegressor(**LGB_PARAMS)
        self.xgb = xgb.XGBRegressor(**XGB_PARAMS)
        self.meta: dict = {}

    def fit(self, df: pd.DataFrame) -> "Ensemble":
        d = df.dropna(subset=[self.target])
        X, y = d[self.features], d[self.target]
        self.lgb.fit(X, y)
        self.xgb.fit(X, y)
        self.meta.update(target=self.target, features=self.features, n_rows=len(d),
                         train_start=str(d["ts"].min()), train_end=str(d["ts"].max()))
        return self

    def predict_all(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = df[self.features]
        p = {"lgbm": self.lgb.predict(X), "xgb": self.xgb.predict(X)}
        p["ensemble"] = (p["lgbm"] + p["xgb"]) / 2
        p["naive"] = df[f"{self.target}_lag168"].to_numpy()
        if self.target == "pv":
            dark = (df["shortwave_radiation"].fillna(0) <= 0).to_numpy()
            for k in p:
                p[k] = np.where(dark, 0.0, np.clip(p[k], 0, None))
        return p

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.lgb.booster_.save_model(folder / f"{self.target}_lgbm.txt")
        self.xgb.save_model(folder / f"{self.target}_xgb.ubj")
        (folder / f"{self.target}_meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, folder: Path, target: str) -> "Ensemble":
        m = cls(target)
        booster = lgb.Booster(model_file=str(folder / f"{target}_lgbm.txt"))
        m.lgb = _BoosterWrapper(booster)
        m.xgb = xgb.XGBRegressor()
        m.xgb.load_model(folder / f"{target}_xgb.ubj")
        m.meta = json.loads((folder / f"{target}_meta.json").read_text())
        return m


class _BoosterWrapper:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(X)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    mae = float(np.mean(np.abs(y - p)))
    return {"mae": mae, "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "wape": float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-9)), "n": int(ok.sum())}
