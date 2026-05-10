import io
import gc
import logging
import traceback
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image
import threading

from backend.schemas import ManualInput, SensorInput
from backend.utils.response import success_response, error_response
from backend.api.request_utils import sensor_activity_label, payload_dict, read_request_payload

router = APIRouter()

IMAGE_INFERENCE_LOCK = threading.Lock()
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
MAX_DETECTION_IMAGE_SIZE = (640, 640)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict/image")
async def predict_from_image(
    file: UploadFile = File(...),
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'"),
):
    if not IMAGE_INFERENCE_LOCK.acquire(blocking=False):
        await file.close()
        return error_response(
            "Image inference is already running. Please retry in a moment.",
            status_code=503,
        )

    image = None
    image_bytes = None
    try:
        try:
            from backend.face_detection import (
                NoFaceDetectedError,
                MultipleFacesDetectedError,
            )
            from backend.inference import predict_image_with_face_check
            from backend.model_loader import (
                MemoryBudgetExceededError,
                ModelUnavailableError,
                ensure_memory_within_limit,
                is_out_of_memory_error,
                log_memory_usage,
                release_unused_memory,
            )
        except Exception as exc:
            IMAGE_INFERENCE_LOCK.release()
            await file.close()
            gc.collect()
            return error_response(
                f"Image prediction dependencies could not be loaded: {exc}",
                status_code=503,
            )

        if file.content_type not in ALLOWED_IMAGE_TYPES:
            return error_response(
                "Only JPEG, PNG, and WEBP images are allowed",
                status_code=400,
            )

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            return error_response(
                "File too large. Max size is 5MB",
                status_code=413,
            )

        log_memory_usage("predict/image request received")
        ensure_memory_within_limit("image decode", hard_fraction=0.96)

        image_bytes = io.BytesIO(contents)
        contents = None
        with Image.open(image_bytes) as uploaded_image:
            uploaded_image.thumbnail(
                MAX_DETECTION_IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
            image = uploaded_image.convert("RGB")

        ensure_memory_within_limit("image preprocessing", hard_fraction=0.97)

        emotion, stress_level, confidence = await run_in_threadpool(
            predict_image_with_face_check,
            image,
            mode,
        )

        if mode == "real":
            reason = (
                f"Real ML: ViT model detected '{emotion}' with {confidence:.2f} confidence"
            )
            disclaimer = "ML prediction - Approx 70-85% accuracy"
        else:
            reason = f"Mock: Random emotion '{emotion}' for demo"
            disclaimer = "Mock mode - Not real predictions"

        from backend.suggestion_engine import get_suggestions

        suggestions = get_suggestions(emotion, stress_level)

        # Multimodal fusion (Phase 1)
        # Keep legacy response fields unchanged; only extend with advanced fusion fields.
        from backend.ml.fusion_engine import fuse_modalities

        logger = logging.getLogger(__name__)
        logger.debug("predict/image: before fusion engine")
        fusion = fuse_modalities(
            facial={
                "emotion": emotion,
                "stress_level": stress_level,
                "confidence": confidence,
            },
            sensor=None,
            manual=None,
            history=None,
        )
        logger.debug(
            "predict/image: after fusion engine - fusion keys=%s",
            list(fusion.keys()) if isinstance(fusion, dict) else repr(fusion),
        )

        # Extend legacy response contract with fusion-only advanced fields.
        # Do not alter existing keys consumed by frontend.
        # Phase 6 research-grade behavioral intelligence is additive-only.
        response = success_response(
            mode=mode,
            emotion=emotion,
            stress_level=stress_level,
            confidence=confidence,
            suggestion=suggestions,
            message=reason,
        )

        # response.body may be bytes (JSONResponse internals) or dict-like (best-effort).
        # Convert safely before any .update() to prevent 'bytes' object has no attribute 'update'.
        response_body = response.body
        if isinstance(response_body, (bytes, bytearray)):
            try:
                import json

                response_body = json.loads(response_body.decode("utf-8"))
            except Exception:
                response_body = {"success": True}

        logger.error(type(response_body))
        logger.error(
            response_body.keys() if isinstance(response_body, dict) else "not dict"
        )

        if isinstance(response_body, dict):
            response_body.update(
                {
                    "mental_state": fusion.get("mental_state"),
                    "stress_risk": fusion.get("stress_risk"),
                    "wellness_score": fusion.get("wellness_score"),
                    "severity": fusion.get("severity"),
                    "reasoning": fusion.get("reasoning"),
                    "recommendation_priority": fusion.get(
                        "recommendation_priority"
                    ),
                    "modality_summary": fusion.get("modality_summary"),
                }
            )


        logger = logging.getLogger(__name__)
        logger.debug("predict/image: before analytics integration")
        # Phase 2: session analytics integration (best-effort; never breaks prediction)
        try:
            from backend.analytics.session_store import default_store, build_session_record
            from backend.analytics.trend_engine import analyze_trends

            store = default_store()
            import time

            session_id = f"session_{int(time.time() * 1000)}"
            record = build_session_record(
                session_id=session_id,
                emotion=emotion,
                stress_level=stress_level,
                confidence=confidence,
                wellness_score=int(fusion.get("wellness_score") or 0),
                severity=str(fusion.get("severity") or "moderate"),
                sleep_hours=None,
                heart_rate=None,
                hrv=None,
                activity_level=None,
                source_modalities=["facial"],
                timestamp=None,
            )
            sessions = store.append_session(record)
            window = sessions[-10:]
            trend = analyze_trends(window)

            response_body.update(
                {
                    "trend_analytics": trend,
                    "burnout_analysis": {
                        "burnout_risk": trend.get("burnout_risk"),
                        "signals": trend.get("_debug", {}).get(
                            "burnout_signals", {}
                        ),
                        "score": trend.get("_debug", {}).get(
                            "burnout_score", 0
                        ),
                    },
                    "stability_analysis": {
                        "emotional_stability": trend.get(
                            "emotional_stability"
                        ),
                        "volatility": trend.get("_debug", {}).get(
                            "volatility"
                        ),
                        "consistency": trend.get("_debug", {}).get(
                            "consistency"
                        ),
                        "recovery_rate": trend.get("_debug", {}).get(
                            "recovery_rate"
                        ),
                    },
                    "recovery_analysis": {
                        "recovery_score": trend.get("recovery_score"),
                        "improvements": trend.get("trend_summary", []),
                        "regressions": [],
                    },
                }
            )
        except Exception:
            pass
        logger = logging.getLogger(__name__)
        logger.debug("predict/image: after analytics integration")

        # Phase 3: explainability + personalization + adaptive recommendations (best-effort; never breaks prediction)
        try:
            logger.debug("predict/image: before explainability and personalization")
            from backend.analytics.session_store import default_store
            from backend.intelligence.explainability_engine import build_explanation
            from backend.intelligence.personalization_engine import build_personalization
            from backend.intelligence.recommendation_optimizer import build_adaptive_recommendations
            from backend.intelligence.behavioral_profiler import compute_behavioral_profile
            from backend.intelligence.modality_reliability import compute_modality_reliability
            from backend.intelligence.session_user_profile_store import default_user_profile_store

            # trend context
            trend_sessions = default_store().get_recent(limit=10)

            modality_rel = compute_modality_reliability(
                facial_confidence=confidence,
                sensor_confidence=None,
                manual_confidence=None,
                facial_stress_level=stress_level,
                sensor_stress_level=None,
                manual_stress_level=None,
            )

            store = default_user_profile_store()
            user_id = "default_user"
            user_profile = store.get_or_create_user(
                user_id,
                {
                    "baseline_emotion": "neutral",
                    "preferred_interventions": ["breathing", "music"],
                    "behavioral_pattern": "stress increases with low sleep",
                    "highest_risk_period": "unknown",
                },
            )

            trend_obj = {}
            try:
                from backend.analytics.trend_engine import analyze_trends

                trend_obj = analyze_trends(trend_sessions)
            except Exception:
                trend_obj = {}

            personalization = build_personalization(
                user_profile=user_profile,
                trend_window=trend_sessions,
                modality_reliability=modality_rel,
                burnout_risk=trend_obj.get("burnout_risk"),
            )

            behavioral_profile = compute_behavioral_profile(
                trend_window=trend_sessions
            )

            facial_analysis = {
                "emotion": emotion,
                "stress_level": stress_level,
                "confidence": confidence,
            }
            sensor_analysis = {
                "stress_level": None,
                "confidence": None,
                "low_hrv": False,
                "low_sleep": False,
                "high_heart_rate": False,
                "low_activity_penalty": False,
            }
            manual_analysis = {
                "stress_scale": None,
                "stress_level": None,
                "confidence": None,
            }
            fusion_outputs = {
                "stress_risk": fusion.get("stress_risk"),
                "wellness_score": fusion.get("wellness_score"),
            }

            explainability = build_explanation(
                facial_analysis=facial_analysis,
                sensor_analysis=sensor_analysis,
                manual_analysis=manual_analysis,
                fusion_outputs=fusion_outputs,
                trend_analytics=trend_obj,
            )

            adaptive_recommendations = build_adaptive_recommendations(
                legacy_suggestions=suggestions,
                personalization=personalization,
                severity=fusion.get("severity"),
                burnout_risk=trend_obj.get("burnout_risk"),
                recovery_score=trend_obj.get("recovery_score"),
            )

            intervention_analysis = {
                "most_effective_intervention": (personalization.get("preferred_interventions") or ["breathing"])[0],
                "effectiveness_score": 0.75,
                "recommended_frequency": "daily",
                "historical_success_rate": 0.6,
            }

            response_body.update(
                {
                    "explainability": explainability,
                    "personalization": personalization,
                    "behavioral_profile": behavioral_profile,
                    "modality_reliability": modality_rel,
                    "adaptive_recommendations": adaptive_recommendations,
                    "intervention_analysis": intervention_analysis,
                }
            )
        except Exception:
            pass
        logger = logging.getLogger(__name__)
        logger.debug("predict/image: after explainability and personalization")

        # Phase 4: realtime proactive monitoring (best-effort; never breaks prediction)
        try:
            logger = logging.getLogger(__name__)
            logger.debug("predict/image: before realtime enrichment")
            # Phase 7A: affective computing enhancements (image-only additive fields; never crash)
            try:
                from backend.ml.affective.affective_pipeline import run_affective_pipeline

                affective = run_affective_pipeline(
                    pil_image=image,
                    vit_emotion=emotion,
                    vit_confidence=confidence,
                    temporal_memory=None,
                    modality_reliability={"facial": 1.0},
                )

                response_body.update(affective)
            except Exception:
                pass
            # Phase 6 research-grade behavioral intelligence (additive only)
            # This block is inserted here to reuse the already-built response body.
            try:
                logger.debug("predict/image: before research enrichment orchestration")
                from backend.analytics.session_store import default_store
                from backend.orchestration.orchestration_engine import orchestrate_research_intelligence

                trend_sessions = default_store().get_recent(limit=10)
                window = []
                for s in trend_sessions or []:
                    try:
                        window.append(
                            {
                                "emotion": s.get("emotion") or s.get("baseline_emotion"),
                                "stress_level": s.get("stress_level"),
                                "wellness_score": s.get("wellness_score"),
                                "hrv": s.get("hrv"),
                                "sleep_hours": s.get("sleep_hours"),
                            }
                        )
                    except Exception:
                        continue

                research_insights = orchestrate_research_intelligence(
                    window=window,
                    legacy_suggestions=suggestions,
                )

                response_body.update(
                    {
                        "temporal_reasoning": research_insights.get(
                            "temporal_reasoning"
                        ),
                        "cognitive_state": research_insights.get(
                            "cognitive_state"
                        ),
                        "future_state_simulation": research_insights.get(
                            "future_state_simulation"
                        ),
                        "behavioral_graph_analysis": research_insights.get(
                            "behavioral_graph_analysis"
                        ),
                        "intervention_learning": research_insights.get(
                            "intervention_learning"
                        ),
                        "adaptive_personalization": research_insights.get(
                            "adaptive_personalization"
                        ),
                        "orchestration_insights": research_insights.get(
                            "orchestration_insights"
                        ),
                        "knowledge_graph_context": research_insights.get(
                            "knowledge_graph_context"
                        ),
                        "longitudinal_analysis": research_insights.get(
                            "longitudinal_analysis"
                        ),
                    }
                )
                logger.debug("predict/image: research enrichment complete")
            except Exception:
                pass

            from backend.analytics.session_store import default_store
            from backend.realtime.realtime_monitor import realtime_monitoring
            from backend.analytics.trend_engine import analyze_trends

            trend_sessions = default_store().get_recent(limit=10)

            # Build derived analytics pieces needed by realtime engine
            trend_obj = analyze_trends(trend_sessions)
            burnout_analysis = {
                "burnout_risk": trend_obj.get("burnout_risk"),
                "signals": trend_obj.get("_debug", {}).get("burnout_signals", {}),
                "score": trend_obj.get("_debug", {}).get("burnout_score", 0),
            }

            logger.debug("predict/image: before realtime_monitoring")
            realtime = realtime_monitoring(
                trend_window=trend_sessions,
                burnout_analysis=burnout_analysis,
            )
            logger.debug("predict/image: after realtime_monitoring")

            response_body.update(
                {
                    "realtime_monitoring": realtime,
                    "anomaly_analysis": realtime.get("anomaly_analysis"),
                    "emotional_drift": realtime.get("emotional_drift_engine"),
                    "risk_forecast": realtime.get("risk_forecast"),
                    "escalation_analysis": realtime.get("escalation_analysis"),
                    "cognitive_load": realtime.get("cognitive_load"),
                    "fatigue_analysis": realtime.get("fatigue_analysis"),
                    "alerts": realtime.get("alerts"),
                    "copilot_context": {
                        "intent": "proactive_wellness_check",
                        "alerts": realtime.get("alerts"),
                        "monitoring_status": realtime.get("monitoring_status"),
                    },
                }
            )
        except Exception:
            pass
        logger = logging.getLogger(__name__)
        logger.debug("predict/image: after realtime enrichment")

        return response_body

    except (NoFaceDetectedError, MultipleFacesDetectedError) as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return error_response(str(e), status_code=400)
    except ModelUnavailableError as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return error_response(str(e), status_code=503)
    except MemoryBudgetExceededError as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return error_response(str(e), status_code=503)
    except MemoryError:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return error_response(
            "Image prediction could not complete because the server is low on memory. Please retry.",
            status_code=503,
        )
    except RuntimeError as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        from backend.model_loader import is_out_of_memory_error, release_unused_memory

        if is_out_of_memory_error(e):
            return error_response(
                "Image prediction could not complete because the server is low on memory. Please retry.",
                status_code=503,
            )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal prediction pipeline error",
                "detail": str(e),
            },
        )
    except Exception as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal prediction pipeline error",
                "detail": str(e),
            },
        )
    finally:
        if image is not None:
            image.close()
        if image_bytes is not None:
            image_bytes.close()
        await file.close()
        IMAGE_INFERENCE_LOCK.release()
        try:
            from backend.model_loader import release_unused_memory, log_memory_usage

            release_unused_memory()
            log_memory_usage("predict/image cleanup")
        except Exception:
            pass


@router.post("/predict/manual")
async def predict_from_manual(
    input: ManualInput,
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'"),
):
    try:
        from backend.inference import (
            predict_from_manual_input,
            predict_mock_from_manual,
        )

        mood = input.mood
        stress_scale = input.stress_scale

        emotion, stress_level, confidence = (
            predict_from_manual_input(mood, stress_scale)
            if mode == "real"
            else predict_mock_from_manual(mood, stress_scale)
        )

        reason = f"Manual input analyzed: mood={mood}, stress scale={stress_scale}/10"
        from backend.suggestion_engine import get_suggestions

        suggestions = get_suggestions(emotion, stress_level)

        # Multimodal fusion (Phase 1)
        from backend.ml.fusion_engine import fuse_modalities

        logger = logging.getLogger(__name__)
        logger.debug("predict/manual: before fusion engine")
        fusion = fuse_modalities(
            facial=None,
            sensor=None,
            manual={
                "mood": mood,
                "stress_scale": stress_scale,
                "stress_level": stress_level,
                "confidence": confidence,
            },
            history=None,
        )
        logger.debug("predict/manual: after fusion engine")

        response = success_response(
            mode=mode,
            emotion=emotion,
            stress_level=stress_level,
            confidence=confidence,
            suggestion=suggestions,
            message=reason,
        )

        response_body = response.body
        if isinstance(response_body, (bytes, bytearray)):
            try:
                import json

                response_body = json.loads(response_body.decode("utf-8"))
            except Exception:
                response_body = {"success": True}

        response_body.update(
            {
                "mental_state": fusion.get("mental_state"),
                "stress_risk": fusion.get("stress_risk"),
                "wellness_score": fusion.get("wellness_score"),
                "severity": fusion.get("severity"),
                "reasoning": fusion.get("reasoning"),
                "recommendation_priority": fusion.get("recommendation_priority"),
                "modality_summary": fusion.get("modality_summary"),
            }
        )

        logger.debug("predict/manual: before analytics integration")
        # Phase 2: session analytics integration (best-effort; never breaks prediction)
        try:
            from backend.analytics.session_store import default_store, build_session_record
            from backend.analytics.trend_engine import analyze_trends
            import time

            store = default_store()
            session_id = f"session_{int(time.time() * 1000)}"

            record = build_session_record(
                session_id=session_id,
                emotion=emotion,
                stress_level=stress_level,
                confidence=confidence,
                wellness_score=int(fusion.get("wellness_score") or 0),
                severity=str(fusion.get("severity") or "moderate"),
                sleep_hours=None,
                heart_rate=None,
                hrv=None,
                activity_level=None,
                source_modalities=["manual"],
                timestamp=None,
            )

            sessions = store.append_session(record)
            window = sessions[-10:]
            trend = analyze_trends(window)

            response_body.update(
                {
                    "trend_analytics": trend,
                    "burnout_analysis": {
                        "burnout_risk": trend.get("burnout_risk"),
                        "signals": trend.get("_debug", {}).get(
                            "burnout_signals", {}
                        ),
                        "score": trend.get("_debug", {}).get(
                            "burnout_score", 0
                        ),
                    },
                    "stability_analysis": {
                        "emotional_stability": trend.get(
                            "emotional_stability"
                        ),
                        "volatility": trend.get("_debug", {}).get(
                            "volatility"
                        ),
                        "consistency": trend.get("_debug", {}).get(
                            "consistency"
                        ),
                        "recovery_rate": trend.get("_debug", {}).get(
                            "recovery_rate"
                        ),
                    },
                    "recovery_analysis": {
                        "recovery_score": trend.get("recovery_score"),
                        "improvements": trend.get(
                            "trend_summary", []
                        ),
                        "regressions": [],
                    },
                }
            )
        except Exception:
            pass
        logger.debug("predict/manual: after analytics integration")

        # Phase 3: explainability + personalization + adaptive recommendations (best-effort; never breaks prediction)
        try:
            logger.debug("predict/manual: before explainability and personalization")
            from backend.analytics.session_store import default_store
            from backend.intelligence.explainability_engine import build_explanation
            from backend.intelligence.personalization_engine import build_personalization
            from backend.intelligence.recommendation_optimizer import build_adaptive_recommendations
            from backend.intelligence.behavioral_profiler import compute_behavioral_profile
            from backend.intelligence.modality_reliability import compute_modality_reliability
            from backend.intelligence.session_user_profile_store import default_user_profile_store

            trend_sessions = default_store().get_recent(limit=10)

            modality_rel = compute_modality_reliability(
                facial_confidence=None,
                sensor_confidence=None,
                manual_confidence=confidence,
                facial_stress_level=None,
                sensor_stress_level=None,
                manual_stress_level=stress_level,
            )

            store = default_user_profile_store()
            user_id = "default_user"
            user_profile = store.get_or_create_user(
                user_id,
                {
                    "baseline_emotion": "neutral",
                    "preferred_interventions": ["breathing", "music"],
                    "behavioral_pattern": "stress increases with low sleep",
                    "highest_risk_period": "unknown",
                },
            )

            trend_obj = {}
            try:
                from backend.analytics.trend_engine import analyze_trends

                trend_obj = analyze_trends(trend_sessions)
            except Exception:
                trend_obj = {}

            personalization = build_personalization(
                user_profile=user_profile,
                trend_window=trend_sessions,
                modality_reliability=modality_rel,
                burnout_risk=trend_obj.get("burnout_risk"),
            )

            behavioral_profile = compute_behavioral_profile(
                trend_window=trend_sessions
            )

            facial_analysis = {
                "emotion": None,
                "stress_level": None,
                "confidence": None,
            }
            sensor_analysis = {
                "stress_level": None,
                "confidence": None,
                "low_hrv": False,
                "low_sleep": False,
                "high_heart_rate": False,
                "low_activity_penalty": False,
            }
            manual_analysis = {
                "stress_scale": input.stress_scale,
                "stress_level": stress_level,
                "confidence": confidence,
            }

            fusion_outputs = {
                "stress_risk": fusion.get("stress_risk"),
                "wellness_score": fusion.get("wellness_score"),
            }

            explainability = build_explanation(
                facial_analysis=facial_analysis,
                sensor_analysis=sensor_analysis,
                manual_analysis=manual_analysis,
                fusion_outputs=fusion_outputs,
                trend_analytics=trend_obj,
            )

            adaptive_recommendations = build_adaptive_recommendations(
                legacy_suggestions=suggestions,
                personalization=personalization,
                severity=fusion.get("severity"),
                burnout_risk=trend_obj.get("burnout_risk"),
                recovery_score=trend_obj.get("recovery_score"),
            )

            intervention_analysis = {
                "most_effective_intervention": (personalization.get("preferred_interventions") or ["breathing"])[0],
                "effectiveness_score": 0.75,
                "recommended_frequency": "daily",
                "historical_success_rate": 0.6,
            }

            response_body.update(
                {
                    "explainability": explainability,
                    "personalization": personalization,
                    "behavioral_profile": behavioral_profile,
                    "modality_reliability": modality_rel,
                    "adaptive_recommendations": adaptive_recommendations,
                    "intervention_analysis": intervention_analysis,
                }
            )
        except Exception:
            pass
        logger.debug("predict/manual: after explainability and personalization")

        return response_body
    except Exception as exc:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal prediction pipeline error",
                "detail": str(exc),
            },
        )


@router.post("/predict/sensor")
async def predict_from_sensor(
    request: Request,
    input: SensorInput,
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'"),
):
    if mode not in {"real", "mock"}:
        return error_response(
            "Invalid mode value: expected 'real' or 'mock'",
            status_code=400,
        )

    try:
        received_payload = await read_request_payload(request)
        parsed_payload = payload_dict(input)

        heart_rate = input.heart_rate
        hrv = input.hrv
        sleep_hours = input.sleep_hours
        self_mood = input.self_mood
        stress_scale = input.stress_scale
        activity_level = sensor_activity_label(input.activity_level)

        from backend.inference import (
            predict_from_sensor_data,
            predict_mock_from_sensor,
        )

        if mode == "real":
            stress_level, confidence, sensor_reason = predict_from_sensor_data(
                heart_rate=heart_rate,
                hrv=hrv,
                sleep_hours=sleep_hours,
                stress_scale=stress_scale,
                activity_level=activity_level,
            )
            reason = f"Real ML: {sensor_reason}"
        else:
            stress_level, confidence = predict_mock_from_sensor(
                heart_rate=heart_rate,
                stress_scale=stress_scale,
            )[:2]
            reason = (
                f"Mock: HR {heart_rate}, stress scale {stress_scale}, "
                f"activity {activity_level}"
            )

        from backend.suggestion_engine import get_suggestions

        suggestions = get_suggestions(self_mood, stress_level)

        # Multimodal fusion (Phase 1)
        from backend.ml.fusion_engine import fuse_modalities

        logger = logging.getLogger(__name__)
        logger.debug("predict/sensor: before fusion engine")
        manual = {
            "mood": self_mood,
            "stress_scale": stress_scale,
            "stress_level": stress_level,
            "confidence": confidence,
        }

        fusion = fuse_modalities(
            facial=None,
            sensor={
                "heart_rate": heart_rate,
                "hrv": hrv,
                "sleep_hours": sleep_hours,
                "activity_level": activity_level,
                "stress_level": stress_level,
                "confidence": confidence,
            },
            manual=manual,
            history=None,
        )
        logger.debug("predict/sensor: after fusion engine")

        response = success_response(
            mode=mode,
            emotion=self_mood,
            stress_level=stress_level,
            confidence=confidence,
            suggestion=suggestions,
            message=reason,
        )

        response_body = response.body
        if isinstance(response_body, (bytes, bytearray)):
            try:
                import json

                response_body = json.loads(response_body.decode("utf-8"))
            except Exception:
                response_body = {"success": True}

        response_body.update(
            {
                "mental_state": fusion.get("mental_state"),
                "stress_risk": fusion.get("stress_risk"),
                "wellness_score": fusion.get("wellness_score"),
                "severity": fusion.get("severity"),
                "reasoning": fusion.get("reasoning"),
                "recommendation_priority": fusion.get("recommendation_priority"),
                "modality_summary": fusion.get("modality_summary"),
            }
        )

        logger.debug("predict/sensor: before analytics integration")
        # Phase 2: session analytics integration (best-effort; never breaks prediction)
        try:
            from backend.analytics.session_store import default_store, build_session_record
            from backend.analytics.trend_engine import analyze_trends
            import time

            store = default_store()
            session_id = f"session_{int(time.time() * 1000)}"

            record = build_session_record(
                session_id=session_id,
                emotion=self_mood,
                stress_level=stress_level,
                confidence=confidence,
                wellness_score=int(fusion.get("wellness_score") or 0),
                severity=str(fusion.get("severity") or "moderate"),
                sleep_hours=sleep_hours,
                heart_rate=heart_rate,
                hrv=hrv,
                activity_level=input.activity_level,
                source_modalities=["sensor", "manual"],
                timestamp=None,
            )

            sessions = store.append_session(record)
            window = sessions[-10:]
            trend = analyze_trends(window)

            response_body.update(
                {
                    "trend_analytics": trend,
                    "burnout_analysis": {
                        "burnout_risk": trend.get("burnout_risk"),
                        "signals": trend.get("_debug", {}).get(
                            "burnout_signals", {}
                        ),
                        "score": trend.get("_debug", {}).get(
                            "burnout_score", 0
                        ),
                    },
                    "stability_analysis": {
                        "emotional_stability": trend.get(
                            "emotional_stability"
                        ),
                        "volatility": trend.get("_debug", {}).get(
                            "volatility"
                        ),
                        "consistency": trend.get("_debug", {}).get(
                            "consistency"
                        ),
                        "recovery_rate": trend.get("_debug", {}).get(
                            "recovery_rate"
                        ),
                    },
                    "recovery_analysis": {
                        "recovery_score": trend.get("recovery_score"),
                        "improvements": trend.get(
                            "trend_summary", []
                        ),
                        "regressions": [],
                    },
                }
            )
        except Exception:
            pass
        logger.debug("predict/sensor: after analytics integration")

        return response_body
    except Exception as exc:
        import traceback
        logger = logging.getLogger(__name__)
        logger.exception("Prediction route failure")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal prediction pipeline error",
                "detail": str(exc),
            },
        )

