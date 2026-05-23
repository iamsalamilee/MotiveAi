from fastapi import FastAPI
from loguru import logger
from src.app.routes import simulate, recommend
from src.app.models.schemas import HealthResponse

app = FastAPI(
    title="MotiveAi",
    description="Culturally-grounded LLM agent for user simulation and cross-domain recommendation",
    version="1.0.0",
)

app.include_router(simulate.router)
app.include_router(recommend.router)

_startup_done = False

@app.on_event("startup")
async def startup():
    global _startup_done
    logger.info("Starting MotiveAi...")
    # TODO Day 2+: uncomment and wire real components
    # from src.nlp.naija_bert import naija_bert
    # from src.agent.retriever import retriever
    # naija_bert.warmup()
    # retriever.connect(settings.chroma_host, settings.chroma_port)
    _startup_done = True
    logger.info("Startup complete.")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=_startup_done,
        chroma_ready=True,      # TODO: retriever.is_ready()
        items_indexed=0,        # TODO: retriever.get_collection_size()
    )
