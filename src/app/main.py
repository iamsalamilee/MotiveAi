from fastapi import FastAPI
from loguru import logger
from src.app.routes import simulate, recommend
from src.app.models.schemas import HealthResponse
from src.app.core.config import settings

app = FastAPI(
    title="MotiveAi",
    description="Culturally-grounded LLM agent for user simulation and cross-domain recommendation",
    version="1.0.0",
)

app.include_router(simulate.router)
app.include_router(recommend.router)

_startup_done = False
_chroma_ready = False

@app.on_event("startup")
async def startup():
    global _startup_done, _chroma_ready
    logger.info("Starting MotiveAi...")

    # Load NaijaBERT embedder into memory
    try:
        from src.nlp.naija_bert import naija_bert
        naija_bert.warmup()
        logger.info("NaijaBERT loaded successfully.")
    except Exception as e:
        logger.warning(f"NaijaBERT failed to load (will lazy-load on first request): {e}")

    # Connect to ChromaDB
    try:
        from src.agent.retriever import retriever
        import subprocess
        
        retriever.connect(settings.chroma_host, settings.chroma_port)
        _chroma_ready = retriever.is_ready()
        count = retriever.get_collection_size()
        logger.info(f"ChromaDB connected. Items indexed: {count}")
        
        if count == 0:
            logger.warning("ChromaDB empty — running index population...")
            # Run the indexer script automatically using the 4K normalized Yelp data!
            subprocess.Popen([
                "python", "data/build_chroma_index.py",
                "--input", "data/yelp_businesses_normalized.jsonl",
                "--chroma-host", settings.chroma_host,
                "--chroma-port", str(settings.chroma_port),
                "--batch-size", "32"
            ])
            logger.info("Indexing started in background.")
            
    except Exception as e:
        logger.warning(f"ChromaDB connection failed (will retry on first request): {e}")
        _chroma_ready = False

    _startup_done = True
    logger.info("Startup complete.")

@app.get("/health", response_model=HealthResponse)
async def health():
    from src.agent.retriever import retriever
    return HealthResponse(
        status="ok",
        models_loaded=_startup_done,
        chroma_ready=retriever.is_ready() if _startup_done else False,
        items_indexed=retriever.get_collection_size() if _startup_done else 0,
    )

