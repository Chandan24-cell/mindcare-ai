from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer
import datetime
import os

def generate_report(email, emotion, stress_level, confidence, suggestions, reason=None):
    """
    Generate a professional PDF report with analysis results.

    Args:
        email: User's email address
        emotion: Detected emotion
        stress_level: Calculated stress level (low/medium/high)
        confidence: Model confidence score (0-1) as percentage string
        suggestions: List of wellness recommendations
        reason: Detailed analysis explanation (optional)
    """
    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)

    # Setup canvas
    width, height = letter
    c = canvas.Canvas(filepath, pagesize=letter)
    margin = 0.75 * inch  # 0.75 inch margins

    # Title - Centered
    c.setFont("Helvetica-Bold", 18)
    title = "MindCare AI – Mental State Analysis Report"
    title_width = c.stringWidth(title, "Helvetica-Bold", 18)
    c.drawString((width - title_width) / 2, height - margin, title)

    # Draw decorative line under title
    c.setStrokeColor((0.2, 0.6, 0.8))
    c.setLineWidth(1)
    c.line(margin, height - margin - 10, width - margin, height - margin - 10)

    # Start content block
    y = height - margin - 40
    line_height = 20
    section_spacing = 30

    # Styles
    label_style = "Helvetica-Bold"
    label_size = 12
    value_style = "Helvetica"
    value_size = 12

    c.setFont(label_style, label_size)

    # User Info
    c.drawString(margin, y, "User:")
    c.setFont(value_style, value_size)
    c.drawString(margin + 60, y, email)
    y -= line_height

    c.setFont(label_style, label_size)
    c.drawString(margin, y, "Date:")
    c.setFont(value_style, value_size)
    date_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    c.drawString(margin + 60, y, date_str)
    y -= section_spacing

    # Results Section Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Analysis Results")
    y -= section_spacing

    # Emotion
    c.setFont(label_style, label_size)
    c.drawString(margin, y, "Detected Emotion:")
    c.setFont(value_style, value_size)
    c.drawString(margin + 120, y, emotion.title())
    y -= line_height

    # Stress Level
    c.setFont(label_style, label_size)
    c.drawString(margin, y, "Stress Level:")
    c.setFont(value_style, value_size)
    c.drawString(margin + 120, y, stress_level.title())
    y -= line_height

    # Confidence
    c.setFont(label_style, label_size)
    c.drawString(margin, y, "Confidence Score:")
    c.setFont(value_style, value_size)
    # Format confidence as percentage if it's a number
    if isinstance(confidence, (int, float)):
        conf_str = f"{confidence * 100:.1f}%"
    else:
        conf_str = str(confidence)
    c.drawString(margin + 120, y, conf_str)
    y -= section_spacing

    # Detailed Analysis (if provided)
    if reason:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Detailed Analysis:")
        y -= line_height
        c.setFont("Helvetica", 10)

        # Wrap text to fit width
        from textwrap import wrap
        reason_text = str(reason)
        max_width = width - 2 * margin
        lines = wrap(reason_text, width=int(max_width / 6))  # Approx chars per line
        for line in lines:
            if y < margin + 40:  # Check for page overflow (simple)
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= 15
        y -= 10

    # Recommendations Section
    if suggestions and len(suggestions) > 0:
        # Ensure we have space
        if y < margin + 100:
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Personalized Recommendations:")
        y -= line_height

        c.setFont("Helvetica", 10)
        for i, suggestion in enumerate(suggestions, 1):
            if y < margin + 40:
                c.showPage()
                y = height - margin
            # Bullet point with number
            bullet = f"{i}. {suggestion}"
            # Wrap long suggestions
            from textwrap import wrap
            lines = wrap(bullet, width=int((width - 2 * margin) / 6))
            for j, line in enumerate(lines):
                if j == 0:
                    c.drawString(margin + 20, y, line)
                else:
                    c.drawString(margin + 40, y, line)
                y -= 15
            y -= 5  # Space between items

    # Footer with disclaimer
    c.setFont("Helvetica-Oblique", 8)
    footer_text = "This report is for informational purposes only and does not constitute medical advice. Please consult a qualified healthcare professional for medical concerns."
    footer_y = margin / 2
    c.drawString(margin, footer_y, footer_text)

    c.save()
    return filepath
