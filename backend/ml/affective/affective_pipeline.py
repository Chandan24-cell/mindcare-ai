from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PIL import Image

from backend.ml.affective.adaptive_thresholds import get_default_thresholds
from backend.ml.affective.emotion_memory_buffer import EmotionMemoryBuffer
from backend.ml.affective.face_mesh_engine import estimate_face_mesh_metrics
from backend.ml.affective.facial_signal_extractor import extract_facial_signals
from backend.ml.affective.micro_expression_engine import analyze_micro_expression
from backend.ml.affective.emotion_stabilizer import stabilize_emotion
from backend.ml.affective.confidence_calibrator import confidence_calibration
from backend.ml.affective.multimodel_consensus import build_model_consensus
from backend.ml.affective.reliability_estimator import estimate_reliability


logger = logging.getLogger(__name__)


def run_affective_pipeline(
    *,
    pil_image: Image.Image,
    vit_emotion: str,
    vit_confidence: float,
    temporal_memory: Optional[EmotionMemoryBuffer] = None,
    modality_reliability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the image-only affective pipeline.

    Deterministic, CPU-safe, and defensive.
    """

    thresholds = get_default_thresholds()
    memory = temporal_memory or EmotionMemoryBuffer(maxlen=5)

    # 1) Face landmarks
    face_landmark_metrics = estimate_face_mesh_metrics(pil_image)
    face_landmark_analysis = {
        "eye_openness": face_landmark_metrics.eye_openness,
        "mouth_tension": face_landmark_metrics.mouth_tension,
        "brow_stress": face_landmark_metrics.brow_stress,
        "head_tilt": face_landmark_metrics.head_tilt,
        "facial_symmetry": face_landmark_metrics.facial_symmetry,
        "landmark_confidence": face_landmark_metrics.landmark_confidence,
    }

    # 2) Facial signals
    facial_signals = extract_facial_signals(landmark_metrics=face_landmark_metrics)

    # 3) Temporal stabilization (buffer over last predictions)
    temporal_output = stabilize_emotion(
        current_emotion=vit_emotion,
        current_confidence=vit_confidence,
        memory=memory,
    )

    stabilized_emotion = temporal_output.get("stabilized_emotion", vit_emotion)
    temporal_stability = {
        "stabilized_emotion": stabilized_emotion,
        "stability_score": temporal_output.get("stability_score", 0.5),
        "emotion_variance": temporal_output.get("emotion_variance", 0.0),
    }

    # 4) Micro-expression
    micro_expression_analysis = analyze_micro_expression(
        stabilized_emotion=stabilized_emotion,
        landmark_metrics=face_landmark_analysis,
    )

    # 5) Confidence calibration
    uncertainty_score = 0.0
    try:
        uncertainty_score = float(micro_expression_analysis.get("hidden_stress_probability", 0.0))
    except Exception:
        uncertainty_score = 0.0

    reliability_score_guess = 0.6
    confidence_output = confidence_calibration(
        raw_confidence=vit_confidence,
        uncertainty_score=uncertainty_score,
        reliability_score=reliability_score_guess,
        modality_reliability=modality_reliability,
    )

    # 6) Reliability estimation
    reliability_output = estimate_reliability(
        raw_confidence=vit_confidence,
        face_landmark_confidence=face_landmark_metrics.landmark_confidence,
        temporal_stability_score=temporal_output.get("stability_score"),
        quality_flags=[],
    )

    # Re-calibrate with estimated reliability
    confidence_output = confidence_calibration(
        raw_confidence=vit_confidence,
        uncertainty_score=uncertainty_score,
        reliability_score=float(reliability_output.get("reliability_score", 0.6)),
        modality_reliability=modality_reliability,
    )

    # 7) Multi-model consensus
    model_consensus = build_model_consensus(
        vit_emotion=vit_emotion,
        vit_confidence=vit_confidence,
        stabilized_emotion=stabilized_emotion,
        calibrated_confidence=float(confidence_output.get("calibrated_confidence", vit_confidence)),
        facial_signal_confidence=float(facial_signals.get("engagement_signal", 0.5)),
    )

    affective_analysis = {
        "consensus_emotion": model_consensus.get("consensus_emotion"),
        "consensus_confidence": model_consensus.get("consensus_confidence"),
        "agreement_score": model_consensus.get("agreement_score"),
        "stabilized_emotion": stabilized_emotion,
    }

    return {
        "affective_analysis": affective_analysis,
        "facial_signals": facial_signals,
        "temporal_stability": temporal_stability,
        "confidence_calibration": confidence_output,
        "model_consensus": model_consensus,
        "prediction_reliability": reliability_output,
        "micro_expression_analysis": micro_expression_analysis,
        "face_landmark_analysis": face_landmark_analysis,
    }

