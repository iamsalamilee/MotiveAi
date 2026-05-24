from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    model_path: str = "/app/data/models"
    llm_quantize: str = "none"
    log_level: str = "INFO"
    max_history_length: int = 50
    naija_bert_model: str = "airesearch/wangchanberta-base-att-spm-uncased"
    pidgin_vader_path: str = "data/lexicons/pidgin_vader.json"
    pidgin_canonical_path: str = "data/lexicons/pidgin_canonical.json"

    class Config:
        env_file = ".env"

settings = Settings()
