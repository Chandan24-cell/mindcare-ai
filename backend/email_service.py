import os
import smtplib
from pathlib import Path
from dotenv import load_dotenv
from email.message import EmailMessage


def send_report_email(receiver_email: str, pdf_path: str):
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)

    MAIL_SENDER = os.getenv("MAIL_SENDER")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

    print(".env path =", env_path)
    print("MAIL_SENDER =", MAIL_SENDER)
    print("MAIL_PASSWORD =", MAIL_PASSWORD)
    print("RECEIVER =", receiver_email)
    print("PDF EXISTS =", os.path.exists(pdf_path))
    print("PDF PATH =", pdf_path)

    msg = EmailMessage()
    msg["Subject"] = "MindCare AI Report"
    msg["From"] = MAIL_SENDER
    msg["To"] = receiver_email

    msg.set_content(
        f"Hello,\n\n"
        "Thank you for using MindCare AI.\n\n"
        "Your personalized wellness analysis report has been successfully generated and attached to this email.\n\n"
        "This report includes:\n"
        "• Emotion analysis\n"
        "• Stress assessment\n"
        "• AI-powered wellness recommendations\n\n"
        "Your data is processed securely and privately.\n\n"
        "We hope these insights help you maintain a healthier and more balanced lifestyle.\n\n"
        "Take care of yourself,\n\n"
        "MindCare AI\n"
        "AI Mental Wellness Assistant"
    )

    with open(pdf_path, "rb") as f:
        file_data = f.read()

    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="pdf",
        filename="MindCare_Report.pdf",
    )

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)

        server.ehlo()

        server.starttls()

        server.ehlo()

        server.login(MAIL_SENDER, MAIL_PASSWORD)

        server.send_message(msg)

        server.quit()

        print("EMAIL SENT SUCCESSFULLY")

    except Exception as e:
        print("EMAIL ERROR:", str(e))
        raise