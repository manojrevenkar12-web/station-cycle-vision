"""Experiment tracking with MLflow (SQLite backend under runs/<profile>/), degrading to a JSON log
when MLflow is not installed (the edge image does not ship it)."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from typing import Any

from . import config


class Tracking:
    def __init__(self, profile: str) -> None:
        self.dir = config.ROOT / "runs" / profile
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, dict[str, Any]] = {}
        self.ids: dict[str, str] = {}
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{self.dir / 'mlflow.db'}")
            exp = f"station-cycle-vision-{profile}"
            if mlflow.get_experiment_by_name(exp) is None:
                mlflow.create_experiment(exp, artifact_location=(self.dir / "mlartifacts").as_uri())
            mlflow.set_experiment(exp)
            self.mlflow = mlflow
        except Exception:  # noqa: BLE001 - tracking must never stop training
            self.mlflow = None

    @contextlib.contextmanager
    def run(self, name: str, params: dict[str, Any]) -> Iterator[None]:
        flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}
        self.runs[name] = {"params": flat, "metrics": {}}
        if self.mlflow is None:
            yield
            return
        with self.mlflow.start_run(run_name=name) as r:
            self.mlflow.log_params(flat)
            self.ids[name] = r.info.run_id
            yield

    def log_metrics(self, name: str, metrics: dict[str, float]) -> None:
        clean = {k.replace(" ", "_").replace("(", "").replace(")", ""): float(v) for k, v in metrics.items() if v == v}
        self.runs.setdefault(name, {"params": {}, "metrics": {}})["metrics"].update(clean)
        if self.mlflow is not None and name in self.ids:
            with self.mlflow.start_run(run_id=self.ids[name]):
                self.mlflow.log_metrics(clean)

    def summary(self) -> dict[str, Any]:
        return {"backend": "mlflow" if self.mlflow is not None else "json", "runs": sorted(self.runs), "run_ids": self.ids}
