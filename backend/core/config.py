from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    port: int = 7860
    host: str = "127.0.0.1"

    # Mail
    mail_sender: str | None = None
    mail_password: str | None = None

    # OpenRouter / AI enhancements
    openrouter_api_key: str | None = None


settings = Settings()

