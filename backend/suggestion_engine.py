# =============================================================================
# Suggestion Engine Module
# =============================================================================
# This module provides wellness recommendations based on detected emotions
# and stress levels.
#
# Why a Separate Module:
# - Separation of concerns (ML predictions vs business logic)
# - Easy to update recommendations without touching ML code
# - Can be extended with more sophisticated recommendation systems
# - Better testability - can test suggestions independently
# =============================================================================

import os
import logging
import socket
from typing import List, Dict

logger = logging.getLogger(__name__)

# Optional OpenAI integration for AI-powered recommendations
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# Suggestion Database
# =============================================================================
# Maps emotions to baseline wellness recommendations.
# These are evidence-based suggestions for mental wellness.
# =============================================================================

EMOTION_SUGGESTIONS: Dict[str, List[str]] = {
    # Positive emotional states
    "happy": [
        "Maintain your routine and positive habits",
        "Practice gratitude journaling",
        "Engage in light productivity activities",
        "Share your positivity with others"
    ],
    
    # Negative emotional states
    "sad": [
        "Practice deep breathing exercises",
        "Reach out to a friend or loved one",
        "Go for a short walk outside",
        "Allow yourself to feel - emotions are valid"
    ],
    
    # High arousal negative states
    "angry": [
        "Try guided breathing exercises",
        "Reduce environmental stimuli (noise, clutter)",
        "Consider counting to 10 before responding",
        "Consider professional support if persistent"
    ],
    
    # Fear/anxiety states
    "fear": [
        "Practice calming techniques",
        "Use grounding exercises (5-4-3-2-1 method)",
        "Focus on controlled deep breaths",
        "Seek support if anxiety persists"
    ],
    
    # Neutral state - maintenance suggestions
    "neutral": [
        "Monitor your mood throughout the day",
        "Engage in light exercise",
        "Journal your thoughts",
        "Practice mindfulness briefly"
    ],
    
    # Disgust - aversion states
    "disgust": [
        "Take a break from the source",
        "Practice mindfulness",
        "Change your environment",
        "Focus on things you can control"
    ],
    
    # Surprise - high arousal positive
    "surprise": [
        "Ground yourself in the moment",
        "Process the unexpected event",
        "Take deep breaths",
        "Assess the situation calmly"
    ]
}


# =============================================================================
# Stress Level Modifiers
# =============================================================================
# Additional suggestions based on stress intensity.
# =============================================================================

STRESS_MODIFIERS: Dict[str, List[str]] = {
    "high": [
        "Consider consulting a mental health professional",
        "Prioritize self-care and rest",
        "Reduce workload if possible",
        "Practice stress management techniques daily"
    ],
    "medium": [
        "Regular exercise can help manage stress",
        "Consider meditation or yoga",
        "Maintain regular sleep schedule"
    ],
    "low": [
        "Great time for preventive wellness activities",
        "Maintain healthy habits",
        "Build resilience through self-care"
    ]
}


# =============================================================================
# Main Functions
# =============================================================================

def _get_ai_suggestions(emotion: str, stress_level: str) -> List[str]:
    """
    Generate personalized recommendations using OpenAI API.

    Args:
        emotion: Detected emotion
        stress_level: Stress level

    Returns:
        List of AI-generated suggestions, or empty list if API fails
    """
    if not OPENAI_AVAILABLE:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a mental wellness coach. Provide exactly 3 concise, "
                        "actionable tips based on the user's emotional state and stress level. "
                        "Each tip should be 5-12 words. Respond as a plain list, one per line."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Emotion: {emotion}. Stress level: {stress_level}.",
                },
            ],
            max_tokens=180,
            temperature=0.6,
            timeout=10,
        )
        content = response.choices[0].message.content.strip()

        # Parse suggestions (one per line, removing bullets and numbers)
        suggestions = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading bullets/numbers/dashes
            line = line.lstrip(" 0123456789.-*•")
            if line:
                suggestions.append(line)

        logger.info(f"AI generated {len(suggestions)} suggestions for {emotion}/{stress_level}")
        return suggestions[:3]  # exactly three for consistency
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return []


def _has_internet(timeout: float = 0.75) -> bool:
    """
    Quick connectivity check to decide whether to attempt AI calls.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def _rule_based_recommendations(emotion: str, stress_level: str) -> List[str]:
    """
    Deterministic fallback recommendations when AI is unavailable or offline.
    """
    base_suggestions = EMOTION_SUGGESTIONS.get(
        emotion,
        EMOTION_SUGGESTIONS["neutral"]  # Default to neutral suggestions
    )

    stress_suggestions = STRESS_MODIFIERS.get(stress_level, [])

    combined: List[str] = []

    # Prioritise stress-level guidance first so the output reflects both inputs
    for item in stress_suggestions:
        if item not in combined:
            combined.append(item)

    for item in base_suggestions:
        if item not in combined:
            combined.append(item)

    # Ensure at least three actionable items
    if len(combined) < 3:
        for extra in EMOTION_SUGGESTIONS["neutral"]:
            if extra not in combined:
                combined.append(extra)
            if len(combined) >= 3:
                break

    return combined[:3]


def get_suggestions(emotion: str, stress_level: str) -> List[str]:
    """
    Generate wellness recommendations based on emotion and stress level.

    This function combines:
    1. Emotion-specific suggestions (rule-based)
    2. Stress-level modifiers
    3. Optional AI-powered personalization (if OPENAI_API_KEY is set)

    Args:
        emotion: The detected or reported emotion (e.g., "happy", "sad")
        stress_level: The calculated stress level ("low", "medium", "high")

    Returns:
        List of wellness recommendations

    Example:
        >>> get_suggestions("sad", "high")
        ["Practice deep breathing exercises", "Reach out to a friend...",
         "Consider consulting a professional..."]
    """
    # Try AI-powered suggestions first if enabled and online
    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE and _has_internet():
        ai_suggestions = _get_ai_suggestions(emotion, stress_level)
        if ai_suggestions:
            return ai_suggestions

    # Fallback to rule-based suggestions (always returns 3)
    return _rule_based_recommendations(emotion, stress_level)


def get_suggestions_for_mood_only(mood: str) -> List[str]:
    """
    Get suggestions based only on mood (without stress level).
    Used for simple mood-based recommendations.
    
    Args:
        mood: The mood/emotion to get suggestions for
    
    Returns:
        List of mood-specific suggestions
    """
    return EMOTION_SUGGESTIONS.get(mood, EMOTION_SUGGESTIONS["neutral"])


def calculate_stress_from_scale(scale: int) -> str:
    """
    Convert numeric stress scale (1-10) to categorical stress level.
    
    Args:
        scale: Stress level on 1-10 scale
    
    Returns:
        Categorical stress level: "low", "medium", or "high"
    
    Example:
        >>> calculate_stress_from_scale(3)
        "low"
        >>> calculate_stress_from_scale(8)
        "high"
    """
    if scale < 4:
        return "low"
    elif scale > 7:
        return "high"
    else:
        return "medium"
