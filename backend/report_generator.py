import datetime
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


COLORS = {
    "ink": (0.08, 0.11, 0.18),
    "muted": (0.37, 0.43, 0.52),
    "soft": (0.94, 0.97, 0.99),
    "panel": (0.985, 0.992, 1.0),
    "line": (0.82, 0.88, 0.94),
    "cyan": (0.02, 0.58, 0.72),
    "blue": (0.12, 0.32, 0.72),
    "green": (0.03, 0.61, 0.42),
    "amber": (0.86, 0.48, 0.05),
    "rose": (0.84, 0.18, 0.32),
    "violet": (0.43, 0.26, 0.76),
    "white": (1, 1, 1),
}


def _set_fill(c, color):
    c.setFillColorRGB(*color)


def _set_stroke(c, color):
    c.setStrokeColorRGB(*color)


def _confidence_value(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0

    if value > 1:
        value = value / 100
    return max(0.0, min(value, 1.0))


def _stress_color(stress_level):
    normalized = str(stress_level or "").lower()
    if normalized == "high":
        return COLORS["rose"]
    if normalized == "medium":
        return COLORS["amber"]
    if normalized == "low":
        return COLORS["green"]
    return COLORS["cyan"]


def _emotion_color(emotion):
    normalized = str(emotion or "").lower()
    palette = {
        "happy": COLORS["green"],
        "neutral": COLORS["cyan"],
        "sad": COLORS["blue"],
        "angry": COLORS["rose"],
        "fear": COLORS["violet"],
        "disgust": COLORS["amber"],
        "surprise": COLORS["cyan"],
    }
    return palette.get(normalized, COLORS["cyan"])


def _draw_wrapped_text(c, text, x, y, width, font="Helvetica", size=10, leading=14, color=None):
    if color:
        _set_fill(c, color)
    c.setFont(font, size)
    lines = []
    for paragraph in str(text or "").splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=max(18, int(width / (size * 0.48)))) or [""]
        lines.extend(wrapped)

    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def _draw_footer(c, width, margin, timestamp):
    footer_y = 0.45 * inch
    _set_stroke(c, COLORS["line"])
    c.setLineWidth(0.7)
    c.line(margin, footer_y + 18, width - margin, footer_y + 18)
    _set_fill(c, COLORS["muted"])
    c.setFont("Helvetica", 7.6)
    c.drawString(margin, footer_y, "MindCare AI wellness report")
    c.drawRightString(width - margin, footer_y, timestamp)
    c.setFont("Helvetica-Oblique", 7.2)
    disclaimer = (
        "Informational wellness insights only. This report is not medical advice, "
        "diagnosis, or treatment."
    )
    c.drawCentredString(width / 2, footer_y - 13, disclaimer)


def _draw_header(c, width, height, margin, timestamp):
    header_h = 1.35 * inch
    c.saveState()
    _set_fill(c, COLORS["ink"])
    c.roundRect(margin, height - margin - header_h, width - (2 * margin), header_h, 18, fill=1, stroke=0)
    _set_fill(c, COLORS["cyan"])
    c.roundRect(margin + 0.2 * inch, height - margin - 0.58 * inch, 0.32 * inch, 0.32 * inch, 8, fill=1, stroke=0)
    _set_stroke(c, COLORS["white"])
    c.setLineWidth(1.6)
    x = margin + 0.26 * inch
    y = height - margin - 0.42 * inch
    c.line(x, y, x + 0.07 * inch, y)
    c.line(x + 0.07 * inch, y, x + 0.12 * inch, y - 0.14 * inch)
    c.line(x + 0.12 * inch, y - 0.14 * inch, x + 0.18 * inch, y + 0.12 * inch)
    c.line(x + 0.18 * inch, y + 0.12 * inch, x + 0.26 * inch, y)
    _set_fill(c, COLORS["white"])
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin + 0.65 * inch, height - margin - 0.48 * inch, "MindCare AI")
    c.setFont("Helvetica", 9.5)
    c.drawString(margin + 0.65 * inch, height - margin - 0.73 * inch, "Healthcare-tech mental wellness analysis")
    c.setFont("Helvetica-Bold", 13)
    c.drawRightString(width - margin - 0.24 * inch, height - margin - 0.48 * inch, "AI Wellness Report")
    c.setFont("Helvetica", 8.5)
    c.drawRightString(width - margin - 0.24 * inch, height - margin - 0.73 * inch, timestamp)
    _set_fill(c, COLORS["cyan"])
    c.rect(margin, height - margin - header_h, width - (2 * margin), 0.08 * inch, fill=1, stroke=0)
    c.restoreState()
    return height - margin - header_h - 0.38 * inch


def _draw_section_title(c, title, x, y, accent=COLORS["cyan"]):
    _set_fill(c, accent)
    c.roundRect(x, y - 3, 0.08 * inch, 0.24 * inch, 2, fill=1, stroke=0)
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 0.15 * inch, y, title)
    return y - 0.28 * inch


def _draw_metric_card(c, x, y, w, h, label, value, subtext, accent):
    _set_fill(c, COLORS["panel"])
    _set_stroke(c, COLORS["line"])
    c.setLineWidth(0.8)
    c.roundRect(x, y - h, w, h, 14, fill=1, stroke=1)
    _set_fill(c, accent)
    c.roundRect(x + 0.16 * inch, y - 0.36 * inch, 0.24 * inch, 0.24 * inch, 6, fill=1, stroke=0)
    _set_fill(c, COLORS["muted"])
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(x + 0.52 * inch, y - 0.22 * inch, label.upper())
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x + 0.16 * inch, y - 0.68 * inch, str(value))
    _set_fill(c, COLORS["muted"])
    c.setFont("Helvetica", 8)
    c.drawString(x + 0.16 * inch, y - 0.92 * inch, str(subtext))


def _draw_confidence_bar(c, x, y, w, value):
    track_h = 0.14 * inch
    _set_fill(c, COLORS["soft"])
    c.roundRect(x, y, w, track_h, 5, fill=1, stroke=0)
    _set_fill(c, COLORS["cyan"])
    c.roundRect(x, y, max(0.12 * inch, w * value), track_h, 5, fill=1, stroke=0)
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(x + w, y + 0.22 * inch, f"{value * 100:.1f}%")


def _draw_stress_indicator(c, x, y, w, stress_level):
    segments = [("Low", COLORS["green"]), ("Medium", COLORS["amber"]), ("High", COLORS["rose"])]
    active = str(stress_level or "").lower()
    segment_w = (w - 0.12 * inch) / 3
    for index, (label, color) in enumerate(segments):
        sx = x + index * (segment_w + 0.06 * inch)
        is_active = label.lower() == active
        _set_fill(c, color if is_active else COLORS["soft"])
        _set_stroke(c, color if is_active else COLORS["line"])
        c.roundRect(sx, y, segment_w, 0.22 * inch, 7, fill=1, stroke=1)
        _set_fill(c, COLORS["white"] if is_active else COLORS["muted"])
        c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(sx + segment_w / 2, y + 0.075 * inch, label)


def _draw_recommendation_card(c, index, text, x, y, w):
    _set_fill(c, COLORS["panel"])
    _set_stroke(c, COLORS["line"])
    c.setLineWidth(0.8)
    c.roundRect(x, y - 0.72 * inch, w, 0.72 * inch, 12, fill=1, stroke=1)
    _set_fill(c, COLORS["green"])
    c.circle(x + 0.25 * inch, y - 0.31 * inch, 0.11 * inch, fill=1, stroke=0)
    _set_fill(c, COLORS["white"])
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x + 0.25 * inch, y - 0.34 * inch, str(index))
    return _draw_wrapped_text(
        c,
        text,
        x + 0.48 * inch,
        y - 0.22 * inch,
        w - 0.7 * inch,
        font="Helvetica",
        size=9,
        leading=11,
        color=COLORS["ink"],
    )


def generate_report(email, emotion, stress_level, confidence, suggestions, reason=None):
    """
    Generate a polished healthcare-style PDF report with analysis results.

    The route contract is intentionally unchanged; this function only upgrades
    the visual presentation of the existing export payload.
    """
    import uuid

    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_DIR / filename

    expiration_seconds = 24 * 60 * 60
    for old_file in REPORTS_DIR.glob("report_*.pdf"):
        try:
            file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(old_file.stat().st_mtime)
            if file_age.total_seconds() > expiration_seconds:
                old_file.unlink()
        except Exception:
            continue

    width, height = letter
    margin = 0.56 * inch
    timestamp = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    confidence_score = _confidence_value(confidence)
    emotion_label = str(emotion or "Unknown").title()
    stress_label = str(stress_level or "Unknown").title()
    suggestions = suggestions or []

    c = canvas.Canvas(str(filepath), pagesize=letter)
    c.setTitle("MindCare AI Wellness Report")

    y = _draw_header(c, width, height, margin, timestamp)

    # Profile strip
    _set_fill(c, COLORS["soft"])
    _set_stroke(c, COLORS["line"])
    c.roundRect(margin, y - 0.52 * inch, width - 2 * margin, 0.52 * inch, 12, fill=1, stroke=1)
    _set_fill(c, COLORS["muted"])
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(margin + 0.18 * inch, y - 0.18 * inch, "USER EMAIL")
    c.drawString(width / 2, y - 0.18 * inch, "ANALYSIS TIMESTAMP")
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica", 9.5)
    c.drawString(margin + 0.18 * inch, y - 0.38 * inch, str(email or "user@example.com"))
    c.drawString(width / 2, y - 0.38 * inch, timestamp)
    y -= 0.88 * inch

    # Summary cards
    y = _draw_section_title(c, "Analysis Summary", margin, y)
    card_gap = 0.14 * inch
    card_w = (width - 2 * margin - 2 * card_gap) / 3
    card_h = 1.18 * inch
    _draw_metric_card(c, margin, y, card_w, card_h, "Detected Emotion", emotion_label, "Primary emotional state", _emotion_color(emotion))
    _draw_metric_card(c, margin + card_w + card_gap, y, card_w, card_h, "Stress Level", stress_label, "Current stress signal", _stress_color(stress_level))
    _draw_metric_card(c, margin + 2 * (card_w + card_gap), y, card_w, card_h, "Confidence", f"{confidence_score * 100:.1f}%", "Model certainty", COLORS["cyan"])
    y -= card_h + 0.48 * inch

    # Score visualizations
    y = _draw_section_title(c, "Score Indicators", margin, y, COLORS["violet"])
    left_w = (width - 2 * margin - 0.28 * inch) / 2
    right_x = margin + left_w + 0.28 * inch

    _set_fill(c, COLORS["panel"])
    _set_stroke(c, COLORS["line"])
    c.roundRect(margin, y - 1.02 * inch, left_w, 1.02 * inch, 14, fill=1, stroke=1)
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin + 0.18 * inch, y - 0.24 * inch, "AI confidence visualization")
    _draw_confidence_bar(c, margin + 0.18 * inch, y - 0.7 * inch, left_w - 0.36 * inch, confidence_score)

    _set_fill(c, COLORS["panel"])
    _set_stroke(c, COLORS["line"])
    c.roundRect(right_x, y - 1.02 * inch, left_w, 1.02 * inch, 14, fill=1, stroke=1)
    _set_fill(c, COLORS["ink"])
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x + 0.18 * inch, y - 0.24 * inch, "Stress indicator")
    _draw_stress_indicator(c, right_x + 0.18 * inch, y - 0.68 * inch, left_w - 0.36 * inch, stress_level)
    y -= 1.42 * inch

    # Detailed analysis
    y = _draw_section_title(c, "Detailed Analysis", margin, y, COLORS["blue"])
    _set_fill(c, COLORS["panel"])
    _set_stroke(c, COLORS["line"])
    analysis_h = 1.06 * inch
    c.roundRect(margin, y - analysis_h, width - 2 * margin, analysis_h, 14, fill=1, stroke=1)
    analysis_text = reason or "No detailed analysis was provided for this export."
    _draw_wrapped_text(
        c,
        analysis_text,
        margin + 0.22 * inch,
        y - 0.28 * inch,
        width - 2 * margin - 0.44 * inch,
        font="Helvetica",
        size=9.3,
        leading=12.5,
        color=COLORS["ink"],
    )
    y -= analysis_h + 0.42 * inch

    # Recommendations
    y = _draw_section_title(c, "Personalized Recommendations", margin, y, COLORS["green"])
    if suggestions:
        for index, suggestion in enumerate(suggestions[:6], 1):
            if y < 1.25 * inch:
                _draw_footer(c, width, margin, timestamp)
                c.showPage()
                y = _draw_header(c, width, height, margin, timestamp)
                y = _draw_section_title(c, "Personalized Recommendations", margin, y, COLORS["green"])
            _draw_recommendation_card(c, index, suggestion, margin, y, width - 2 * margin)
            y -= 0.84 * inch
    else:
        _set_fill(c, COLORS["muted"])
        c.setFont("Helvetica", 9.5)
        c.drawString(margin, y, "No recommendations were included with this export.")
        y -= 0.32 * inch

    # Historical trend note, without changing export request shape.
    if y < 1.45 * inch:
        _draw_footer(c, width, margin, timestamp)
        c.showPage()
        y = _draw_header(c, width, height, margin, timestamp)

    y -= 0.1 * inch
    y = _draw_section_title(c, "Historical Trend Summary", margin, y, COLORS["amber"])
    _set_fill(c, COLORS["soft"])
    _set_stroke(c, COLORS["line"])
    c.roundRect(margin, y - 0.58 * inch, width - 2 * margin, 0.58 * inch, 12, fill=1, stroke=1)
    _set_fill(c, COLORS["muted"])
    c.setFont("Helvetica", 8.8)
    c.drawString(
        margin + 0.18 * inch,
        y - 0.34 * inch,
        "This export contains the current analysis snapshot. Session trend charts remain available in the dashboard.",
    )

    _draw_footer(c, width, margin, timestamp)
    c.save()
    return f"reports/{filename}"
