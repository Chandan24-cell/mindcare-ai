from fastapi import APIRouter, HTTPException

from backend.utils.response import success_response

router = APIRouter()


@router.post("/generate-report")
async def generate_pdf_report(request: dict):
    try:
        from backend.report_generator import generate_report
        from backend.email_service import send_report_email
        import os

        user_email = request.get("email", "user@example.com")

        filepath = generate_report(
            email=user_email,
            emotion=request.get("emotion", "Unknown"),
            stress_level=request.get("stress_level", "Unknown"),
            confidence=request.get("confidence", 0.0),
            suggestions=request.get("suggestions") or request.get("suggestion", []),
            reason=request.get("reason"),
        )

        send_report_email(user_email, filepath)

        if os.path.exists(filepath):
            os.remove(filepath)

        return {
            "success": True,
            "message": "Report emailed successfully",
            "data": {
                "emailed": True,
                "download_available": False,
                "report_url": None,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate/send report: {str(e)}",
        )

