from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


EMOTION_TO_VALENCE = {
    "happy": 1.0,
    "surprise": 0.8,
    "neutral": 0.5,
    "sad": 0.2,
    "angry": 0.1,
    "fear": 0.15,
    "disgust": 0.12,
}


def _emotion_valence(emotion: Optional[str]) -> Optional[float]:
    e = str(emotion or "").strip().lower()
    if not e:
        return None
    return EMOTION_TO_VALENCE.get(e)


def compute_stability(window: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """Return (emotional_stability, volatility, consistency, recovery_rate)."""
    if not window:
        return 50, 50, 50, 0

    vals = [_emotion_valence(s.get("emotion")) for s in window]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 50, 50, 50, 0

    # volatility: std deviation mapped to 0..100
    import statistics

    try:
        std = float(statistics.pstdev(vals))
    except Exception:
        std = 0.2

    volatility_score = int(round(max(0.0, min(1.0, std / 0.5)) * 100))

    # consistency: higher when consecutive deltas are small
    deltas = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    avg_delta = sum(deltas) / max(1, len(deltas))
    consistency = int(round(max(0.0, min(1.0, 1.0 - avg_delta / 0.6)) * 100))

    # emotional stability: combine inverse volatility + consistency
    emotional_stability = int(round(0.5 * (100 - volatility_score) + 0.5 * consistency))

    # recovery rate proxy: improvement in valence from older->recent
    recovery_rate = 0
    if len(vals) >= 4:
        first = sum(vals[: len(vals) // 2]) / max(1, len(vals) // 2)
        second = sum(vals[len(vals) // 2 :]) / max(1, len(vals) - len(vals) // 2)
        recovery_rate = int(round(max(0.0, min(1.0, (second - first) / 0.7)) * 100))

    return emotional_stability, volatility_score, consistency, recovery_rate

