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

from typing import List, Dict


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

def get_suggestions(emotion: str, stress_level: str) -> List[str]:
    """
    Generate wellness recommendations based on emotion and stress level.
    
    This function combines:
    1. Emotion-specific suggestions
    2. Stress-level modifiers
    
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
    # Get base suggestions for the emotion
    base_suggestions = EMOTION_SUGGESTIONS.get(
        emotion, 
        EMOTION_SUGGESTIONS["neutral"]  # Default to neutral suggestions
    )
    
    # Get stress-level specific modifiers
    stress_suggestions = STRESS_MODIFIERS.get(stress_level, [])
    
    # Combine and return
    all_suggestions = base_suggestions.copy()
    
    # Add stress-specific suggestions (for high stress)
    if stress_level == "high":
        all_suggestions.extend(stress_suggestions)
    elif stress_level == "low":
        # Add positive reinforcement for low stress
        all_suggestions.extend(stress_suggestions)
    
    return all_suggestions


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

