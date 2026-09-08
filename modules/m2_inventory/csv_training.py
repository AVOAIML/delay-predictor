"""Training entry points for the four CSV-backed M2 hazard classifiers."""

from __future__ import annotations

import argparse
import json
import os
from numbers import Real
from pathlib import Path
from typing import Type

import pandas as pd
from mlflow.models import infer_signature

from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.promotion import gate_metrics, should_promote
from maxxflow_mlops.registry import MLflowRegistry
from m2_inventory.hazard_model_base import BaseInventoryHazardModel, InventoryHazardPyfunc
from m2_inventory.inventory_dataset import (
    DEFAULT_DATASET_PATH,
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_COLUMNS,
    InventoryDatasetBuilder,
    MODEL_INPUT_COLUMNS,
    TemporalDatasetSplit,
)

DEFAULT_ARTIFACT_DIR = Path("artifacts/m2_inventory")
MODULE = "m2_inventory"
CURRENT_ARTIFACT = "current"
NO_DRAFT_COVERAGE_ARTIFACT = "no_draft_coverage"
LOCAL_ARTIFACT_FILES = {
    NO_DRAFT_COVERAGE_ARTIFACT: "logistic-regression-no-draft-coverage.joblib",
}
HAZARD_ARTIFACT_CONTRACT = "inventory_hazard_v1"
HAZARD_ALGORITHMS = frozenset(
    {"logistic-regression", "random-forest", "lightgbm", "xgboost"}
)


def model_classes() -> list[Type[BaseInventoryHazardModel]]:
    """Import lazily so an individual model file has minimal dependencies."""
    from m2_inventory.lightgbm_model import LightGBMInventoryModel
    from m2_inventory.logistic_regression_model import LogisticRegressionInventoryModel
    from m2_inventory.random_forest_model import RandomForestInventoryModel
    from m2_inventory.xgboost_model import XGBoostInventoryModel

    return [
        LogisticRegressionInventoryModel,
        RandomForestInventoryModel,
        LightGBMInventoryModel,
        XGBoostInventoryModel,
    ]


def train_csv_model(
    model_class: Type[BaseInventoryHazardModel],
    dataset_path: str | Path | pd.DataFrame = DEFAULT_DATASET_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_state: int = 42,
    *,
    tenant: str = "demo",
    register: bool = False,
    registry: MLflowRegistry | None = None,
) -> tuple[BaseInventoryHazardModel, dict]:
    builder = InventoryDatasetBuilder()
    snapshots = builder.load(dataset_path)
    supervised = builder.build_supervised_frame(snapshots)
    split = builder.split_for_training(supervised, random_state=random_state)
    return _fit_and_save(
        model_class,
        split,
        artifact_dir,
        random_state,
        tenant=tenant,
        register=register,
        registry=registry,
    )


def _fit_and_save(
    model_class: Type[BaseInventoryHazardModel],
    split: TemporalDatasetSplit,
    artifact_dir: str | Path,
    random_state: int,
    *,
    tenant: str,
    register: bool,
    registry: MLflowRegistry | None,
) -> tuple[BaseInventoryHazardModel, dict]:
    model = model_class(random_state=random_state).fit(split)
    metrics = model.evaluate(split.test)
    floor_decision = should_promote(_promotion_candidate_metrics(metrics), None)
    metrics["quality_floor_passed"] = floor_decision.promote
    destination = Path(artifact_dir) / f"{model.algorithm_name}.joblib"
    model.save(destination)
    metrics["artifact_path"] = str(destination)
    if register:
        registration = _register_candidate(
            model,
            split,
            metrics,
            tenant=tenant,
            registry=registry or MLflowRegistry(),
        )
        metrics.update(registration)
    return model, metrics


def train_all_csv_models(
    dataset_path: str | Path | pd.DataFrame = DEFAULT_DATASET_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_state: int = 42,
    *,
    tenant: str = "demo",
    register: bool = False,
    registry: MLflowRegistry | None = None,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict:
    """Train all candidates and select the most trustworthy final-risk model."""
    builder = InventoryDatasetBuilder()
    snapshots = builder.load(dataset_path)
    supervised = builder.build_supervised_frame(snapshots)
    split = builder.split_for_training(
        supervised,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        random_state=random_state,
    )
    active_registry = (registry or MLflowRegistry()) if register else None
    results = []
    for model_class in model_classes():
        _, metrics = _fit_and_save(
            model_class,
            split,
            artifact_dir,
            random_state,
            tenant=tenant,
            register=register,
            registry=active_registry,
        )
        results.append(metrics)

    # The dashboard consumes 30/60-day probabilities, so select on their mean
    # Brier score first. Weekly ECE and AUC break ties and remain diagnostics.
    def selection_key(row):
        horizon_brier = [
            row[name] for name in ("risk_30d_brier", "risk_60d_brier") if name in row
        ]
        final_brier = sum(horizon_brier) / len(horizon_brier) if horizon_brier else row["weekly_brier"]
        return final_brier, row["weekly_ece"], -row["weekly_auc"]

    eligible = [row for row in results if row["quality_floor_passed"]]
    winner = min(
        eligible or results,
        key=selection_key,
    )
    summary = {
        "selected_model": winner["algorithm"],
        "target_definition": "stockout_flag at t+1 conditional on no stockout at t",
        "split_strategy": "chronological_train_calibration_test",
        "train_fraction": train_fraction,
        "calibration_fraction": calibration_fraction,
        "test_fraction": 1.0 - train_fraction - calibration_fraction,
        "hazard_period_days": 7,
        "models": results,
        "metrics": winner,
        "quality_floor_passed": bool(winner["quality_floor_passed"]),
        "selection_warning": (
            None
            if eligible
            else "All 4 candidates failed the absolute MLflow quality floor; "
            f"{winner['algorithm']} is shown for review only, not a usable model. "
            "Publish will refuse it (the floor cannot be forced) and the current "
            "champion — or no model at all, if none exists yet — keeps serving."
        ),
    }
    if register:
        name = registered_model_name(tenant, MODULE)
        summary.update(
            {
                "registered_name": name,
                "version": winner["version"],
                "champion": _champion_metrics(active_registry, name),
            }
        )
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_horizon_calibration_tables(output_dir, winner)
    return summary


def _write_horizon_calibration_tables(output_dir: Path, metrics: dict) -> None:
    """Persist untouched-test reliability tables for the selected candidate."""
    for horizon in ("30d", "60d"):
        tables = []
        for stage in ("before", "independent", "after", "projection_based"):
            rows = metrics.get(f"risk_{horizon}_calibration_table_{stage}", [])
            tables.extend({"stage": stage, **row} for row in rows)
        pd.DataFrame(tables).to_csv(
            output_dir / f"calibration_{horizon}.csv", index=False
        )


def train_for_configurator(
    dataset_path: str | Path | pd.DataFrame,
    tenant: str,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_state: int = 42,
    register: bool = True,
    source: str = "csv",
    logger=None,
) -> dict:
    """Return the result shape consumed by the existing retraining wizard.

    M2 is one product capability backed by four candidate algorithms.  A single
    wizard run therefore trains and evaluates all four, then exposes the selected
    candidate as the version that can be reviewed by the publication gate.
    """
    if logger is not None:
        logger.log("Preparing inventory snapshots and leakage-safe future labels…")
        logger.log("Training Logistic Regression, Random Forest, LightGBM and XGBoost…")
    summary = train_all_csv_models(
        dataset_path,
        artifact_dir,
        random_state,
        tenant=tenant,
        register=register,
    )
    winner = summary["metrics"]
    if logger is not None:
        if summary["quality_floor_passed"]:
            logger.log(
                f"Selected {summary['selected_model']} from four candidates "
                f"(weekly AUC {winner['weekly_auc']:.3f})."
            )
        else:
            logger.log(
                f"None of the four candidates cleared the quality floor. Best of "
                f"a bad set is {summary['selected_model']} (weekly AUC "
                f"{winner['weekly_auc']:.3f}) — not eligible to publish."
            )
        if summary.get("selection_warning"):
            logger.log(f"Quality gate: {summary['selection_warning']}")
    return {
        "model_type": "Inventory hazard classification",
        "metrics": winner,
        "features": list(FEATURE_COLUMNS),
        "dropped_features": list(EXCLUDED_LEAKAGE_COLUMNS),
        "source": source,
        "params": {
            "random_state": random_state,
            "imbalance_strategy": "SMOTE then Tomek Links",
            "candidate_count": 4,
            "target_definition": "stockout_flag at t+1 conditional on no stockout at t",
            "split_strategy": "chronological_train_calibration_test",
            "hazard_period_days": 7,
        },
        "registered_name": summary.get("registered_name"),
        "version": summary.get("version"),
        "champion": summary.get("champion"),
        "selected_model": summary["selected_model"],
        "models": summary["models"],
        "quality_floor_passed": summary["quality_floor_passed"],
        "selection_warning": summary.get("selection_warning"),
    }


def _register_candidate(
    model: BaseInventoryHazardModel,
    split: TemporalDatasetSplit,
    metrics: dict,
    *,
    tenant: str,
    registry: MLflowRegistry,
) -> dict[str, str]:
    """Log one algorithm run and register it as an unpublished candidate."""
    name = registered_model_name(tenant, MODULE)
    pyfunc_model = InventoryHazardPyfunc(model)
    example = _input_example(split.test)
    output = pyfunc_model.predict(None, example)
    signature = infer_signature(example, output)
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    }
    tags = {
        "tenant": tenant,
        "module": MODULE,
        "data_provenance": "csv",
        "algorithm": model.algorithm_name,
        "artifact_contract": HAZARD_ARTIFACT_CONTRACT,
        "calibration_method": str(model.calibrator.method),
        "horizon_calibration_method": str(
            model.constrained_horizon_calibrator.method
            if model.constrained_horizon_calibrator is not None else "none"
        ),
        "imbalance_strategy": "smote_then_tomek",
        "split_strategy": "chronological_train_calibration_test",
        "quality_floor_passed": str(bool(metrics.get("quality_floor_passed", False))).lower(),
        "auc": f"{metrics['weekly_auc']:.8f}",
        "brier": f"{metrics['weekly_brier']:.8f}",
        "ece": f"{metrics['weekly_ece']:.8f}",
        "risk_30d_brier": f"{metrics.get('risk_30d_brier', 1.0):.8f}",
        "risk_60d_brier": f"{metrics.get('risk_60d_brier', 1.0):.8f}",
    }
    version = registry.log_and_register(
        pyfunc_model,
        name=name,
        params={
            "algo": model.algorithm_name,
            "random_state": model.random_state,
            "calibration": model.calibrator.method,
            "horizon_calibration": (
                model.constrained_horizon_calibrator.method
                if model.constrained_horizon_calibrator is not None else "none"
            ),
            "hazard_period_days": 7,
            "risk_30d_weeks": 4,
            "risk_60d_weeks": 9,
            "imbalance_strategy": "smote_then_tomek",
            "tomek_sampling_strategy": "all",
        },
        metrics=numeric_metrics,
        tags=tags,
        signature=signature,
        input_example=example,
        experiment=name,
    )
    return {"registered_name": name, "version": str(version)}


def _input_example(frame: pd.DataFrame) -> pd.DataFrame:
    return _coerce_model_input(frame).head(3).copy()


def _coerce_model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the exact raw MLflow signature for both examples and serving."""
    example = frame.reindex(columns=MODEL_INPUT_COLUMNS).copy()
    for column in ("snapshot_date", "open_po_deadline"):
        example[column] = pd.to_datetime(example[column], errors="coerce")
    string_columns = [
        "item_id",
        "warehouse_id",
        "item_type",
        "unit_of_measurement",
        "warehouse_type",
        "draft_mo_status",
        "mo_status",
    ]
    for column in string_columns:
        example[column] = example[column].fillna("unknown").astype(str)
    numeric_columns = [
        column
        for column in MODEL_INPUT_COLUMNS
        if column not in string_columns
        and column not in {"snapshot_date", "open_po_deadline", "use_rule_based"}
    ]
    for column in numeric_columns:
        example[column] = pd.to_numeric(example[column], errors="coerce").astype(float)
    example["use_rule_based"] = example["use_rule_based"].astype("boolean").fillna(False).astype(bool)
    return example


def _champion_metrics(registry: MLflowRegistry, name: str) -> dict | None:
    version = registry.get_alias_version(name=name, alias="champion")
    if version is None:
        return None
    try:
        require_hazard_artifact(registry, name, str(version))
    except ValueError:
        return None
    tags = registry.get_alias_tags(name=name, alias="champion")
    return {
        "version": version,
        "algorithm": tags.get("algorithm", "unknown"),
        "auc": float(tags.get("auc", 0.0)),
        "brier": float(tags.get("brier", 1.0)),
        "ece": float(tags.get("ece", 1.0)),
        "risk_30d_brier": float(tags.get("risk_30d_brier", 1.0)),
        "risk_60d_brier": float(tags.get("risk_60d_brier", 1.0)),
    }


def publish(
    tenant: str,
    version: str,
    candidate_metrics: dict,
    force: bool = False,
    *,
    registry: MLflowRegistry | None = None,
) -> dict:
    """Apply M1-style quality gates, then move the M2 champion alias."""
    reg = registry or MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    require_hazard_artifact(reg, name, str(version))
    champion = _champion_metrics(reg, name)
    candidate = _promotion_candidate_metrics(candidate_metrics)
    comparison = None if champion is None else {
        "auc": champion["auc"],
        "brier": champion["brier"],
    }
    decision = should_promote(candidate, comparison, force=force)
    if decision.promote:
        reg.promote(name=name, challenger_version=str(version))
    return {
        "published": decision.promote,
        "reasons": decision.reasons,
        "blocker": decision.blocker,
        "gate_checks": decision.checks,
        "gate_summary": decision.summary,
        "champion_before": champion,
        "candidate": candidate_metrics,
    }


def require_hazard_artifact(
    registry: MLflowRegistry, name: str, version: str
) -> dict[str, str]:
    """Reject any registered version that is not an M2 weekly hazard artifact."""
    tags = registry.get_version_tags(name=name, version=str(version))
    valid = (
        tags.get("module") == MODULE
        and tags.get("algorithm") in HAZARD_ALGORITHMS
        and tags.get("split_strategy") == "chronological_train_calibration_test"
        and tags.get("artifact_contract", HAZARD_ARTIFACT_CONTRACT)
        == HAZARD_ARTIFACT_CONTRACT
    )
    if not valid:
        raise ValueError(
            f"M2 version {name!r} v{version} is not a registered weekly hazard artifact"
        )
    return tags


def _promotion_candidate_metrics(candidate_metrics: dict) -> dict:
    # M2 serves calibrated event probabilities, not a hard 0.5 classification
    # decision. With a rare weekly hazard, majority-class accuracy is maximized
    # by predicting "no event" for everyone and is therefore not a valid floor.
    # The gate still enforces ranking, calibration and Brier skill against the
    # constant-prevalence predictor.
    return gate_metrics({
        "auc": float(candidate_metrics["weekly_auc"]),
        "brier": float(candidate_metrics["weekly_brier"]),
        "ece": float(candidate_metrics["weekly_ece"]),
        "n_test": int(candidate_metrics["test_rows"]),
        "positive_rate": float(candidate_metrics["positive_rate"]),
    })


def load_selected_model(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> BaseInventoryHazardModel:
    """Load the calibrated winner recorded by the latest comparison run."""
    output_dir = Path(artifact_dir)
    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Train the CSV candidates first; missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary["selected_model"]
    return BaseInventoryHazardModel.load(output_dir / f"{selected}.joblib")


def load_local_prediction_model(
    artifact: str = CURRENT_ARTIFACT,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> BaseInventoryHazardModel:
    """Load an allow-listed local artifact without changing production defaulting."""
    if artifact == CURRENT_ARTIFACT:
        return load_selected_model(artifact_dir)
    filename = LOCAL_ARTIFACT_FILES.get(artifact)
    if filename is None:
        allowed = sorted({CURRENT_ARTIFACT, *LOCAL_ARTIFACT_FILES})
        raise ValueError(f"unknown local M2 artifact {artifact!r}; choose one of {allowed}")
    path = Path(artifact_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"experimental M2 artifact is not trained yet; missing {path}"
        )
    return BaseInventoryHazardModel.load(path)


def _score_via_champion(
    tenant: str,
    model_input: pd.DataFrame,
    *,
    registry: MLflowRegistry | None = None,
) -> tuple[pd.DataFrame, str]:
    """Score via the published MLflow champion.

    Used both as the dedicated MLflow scoring path and as the fallback when the
    latest CSV comparison run failed the quality floor — the floor already
    blocks *promotion* (``publish``/``should_promote``), but the CSV-artifact
    scoring paths read straight off disk and previously had no equivalent
    check, so a floor-failing candidate could still reach real predictions.
    Returns (predictions, algorithm_name) since a loaded MLflow pyfunc model
    does not expose ``.algorithm_name`` the way the local wrapper does — the
    name comes from the champion alias's own tags instead.
    """
    reg = registry or MLflowRegistry()
    resolved = reg.resolve_champion_name(tenant=tenant, module=MODULE)
    if resolved is None:
        raise RuntimeError(
            f"latest training run failed the quality floor for tenant {tenant!r} "
            "and no published champion exists to fall back to — train and "
            "publish a passing model first"
        )
    name = resolved[0]
    version = reg.get_alias_version(name=name, alias="champion")
    require_hazard_artifact(reg, name, str(version))
    champion = reg.load_champion(name=name)
    algorithm = reg.get_alias_tags(name=name, alias="champion").get("algorithm", "unknown")
    return champion.predict(model_input), algorithm


def score_latest_csv_snapshots(
    tenant: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    registry: MLflowRegistry | None = None,
) -> pd.DataFrame:
    """Score the latest snapshot per item/warehouse using the selected model.

    Falls back to the published MLflow champion when the latest comparison run
    failed the quality floor, instead of silently serving a candidate the
    publish gate would have refused."""
    builder = InventoryDatasetBuilder()
    snapshots = builder.load(dataset_path)
    latest = (
        snapshots.sort_values("snapshot_date")
        .groupby(["item_id", "warehouse_id"], as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    summary_path = Path(artifact_dir) / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("quality_floor_passed", False):
            predictions, _ = _score_via_champion(
                tenant, _coerce_model_input(latest), registry=registry
            )
            return predictions
    return load_selected_model(artifact_dir).predict_risk(latest)


def score_latest_mlflow_snapshots(
    tenant: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    registry: MLflowRegistry | None = None,
) -> pd.DataFrame:
    """Score latest snapshots through the tenant/global MLflow champion."""
    builder = InventoryDatasetBuilder()
    snapshots = builder.load(dataset_path)
    latest = (
        snapshots.sort_values("snapshot_date")
        .groupby(["item_id", "warehouse_id"], as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    predictions, _ = _score_via_champion(
        tenant, _coerce_model_input(latest), registry=registry
    )
    return predictions


def run_model_cli(model_class: Type[BaseInventoryHazardModel]) -> None:
    # When invoked with ``python -m``, the class initially lives in ``__main__``
    # and would create a non-portable pickle. Resolve its canonical import first.
    canonical_class = next(
        candidate for candidate in model_classes() if candidate.__name__ == model_class.__name__
    )
    parser = argparse.ArgumentParser(description=f"Train {model_class.__name__}")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--mlflow-uri")
    args = parser.parse_args()
    if args.mlflow_uri:
        os.environ["MAXXFLOW_MLFLOW_URI"] = args.mlflow_uri
    _, metrics = train_csv_model(
        canonical_class,
        args.data,
        args.output_dir,
        args.seed,
        tenant=args.tenant,
        register=not args.no_register,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare all M2 hazard models")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mlflow-uri")
    args = parser.parse_args()
    if args.mlflow_uri:
        os.environ["MAXXFLOW_MLFLOW_URI"] = args.mlflow_uri
    register = not args.no_register
    result = train_all_csv_models(
        args.data,
        args.output_dir,
        args.seed,
        tenant=args.tenant,
        register=register,
    )
    if args.publish:
        if not register:
            parser.error("--publish cannot be combined with --no-register")
        result["publication"] = publish(
            args.tenant,
            result["version"],
            result["metrics"],
            force=args.force,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
