"""
Suggestion Engine Module - Hybrid Local-First Recommendations
"""


def get_suggestions(emotion, stress_level):
    """
    Generate wellness suggestions based on emotion and stress level.
    Uses a hybrid system: fast local responses with optional AI enhancement.
    """

    local_suggestions = {
        "happy": [
            "Keep maintaining your positive mindset.",
            "Share your positivity with others.",
            "Stay consistent with healthy habits."
        ],
        "sad": [
            "Take a short mindful break.",
            "Talk to someone you trust.",
            "Try calming music or breathing exercises."
        ],
        "neutral": [
            "Stay hydrated and take screen breaks.",
            "Practice light stretching.",
            "Maintain a balanced routine."
        ],
        "angry": [
            "Pause and take deep breaths.",
            "Avoid impulsive reactions.",
            "Take a short walk to relax."
        ]
    }

    # FAST LOCAL RESPONSE
    suggestions = local_suggestions.get(
        emotion.lower(),
        local_suggestions["neutral"]
    )

    return suggestions
# def report_api_endpoint(request):
#     user_email = request.json.get("email")
#     report_data = request.json.get("data")
#     result = generate_and_send_report(user_email, report_data)
#     return jsonify(result)
