from __future__ import annotations

from fastapi import APIRouter

from backend.research.explainability.multimodal_explainer import build_explanation_trace
from backend.research.benchmarking.inference_benchmarks import (
    run_deterministic_inference_benchmarks,
)
from backend.research.exports.json_exporter import export_json

router = APIRouter()


@router.get("/research/benchmarks")
def benchmarks():
    return {
        "success": True,
        "data": {
            "benchmarks": run_deterministic_inference_benchmarks(),
        },
    }


@router.get("/research/explainability")
def explainability():
    trace = build_explanation_trace(
        emotion="neutral",
        stress_level="medium",
        confidence=0.75,
        modality_importance={"facial": 0.4, "sensor": 0.35, "manual": 0.25},
    )
    return {"success": True, "data": trace}


@router.get("/research/export/json")
def export_json_endpoint():
    payload = {"generated": True, "version": "research-placeholder"}
    return {"success": True, "data": export_json(payload)}


@router.get("/research/export/csv")
def export_csv_endpoint():
    return {"success": True, "data": ""}


@router.get("/research/export/markdown")
def export_md_endpoint():
    return {"success": True, "data": ""}


@router.get("/research/export/latex")
def export_latex_endpoint():
    return {"success": True, "data": ""}


@router.get("/research/simulation")
def simulation_endpoint():
    # Deterministic safe placeholder
    return {
        "success": True,
        "data": {
            "future_projection": {"stress_risk_trend": "stable"},
            "risk_projection": {"risk_level": "medium"},
            "intervention_outcomes": {"expected_benefit": 0.5},
        },
    }

