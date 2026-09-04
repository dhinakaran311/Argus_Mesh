"""
backend/app/main.py
====================
AbuseRing Sentinel — FastAPI Application Entry Point

Startup sequence:
  1. Load settings from .env
  2. Connect Neo4j AuraDB
  3. Connect Qdrant Cloud
  4. Load in-memory DataStore (CSVs)
  5. Load XGBoost ML model + feature index
  6. Initialise GraphService + VectorService (seeds Qdrant if empty)
  7. Register all API routers under /api prefix

Shutdown:
  - Close Neo4j driver
  - Release Qdrant client
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import get_settings, clear_settings_cache
from .db.neo4j import Neo4jClient
from .db.qdrant import QdrantDB
from .db.datastore import DataStore
from .services.ml_service import MLService
from .services.embed_service import EmbedService
from .services.graph_service import GraphService
from .services.vector_service import VectorService

from .api import (
    health,
    dashboard,
    clusters,
    transactions,
    model,
    rag,
    investigate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup & shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all services on startup, tear down on shutdown."""
    # Clear cached settings so hot-reload picks up .env changes immediately
    clear_settings_cache()
    settings = get_settings()
    log.info("=" * 60)
    log.info("  AbuseRing Sentinel — Starting up")
    log.info("=" * 60)

    # -------------------------------------------------------------------------
    # 0. Populate os.environ from pydantic-settings values
    #    Third-party libraries (ChatGroq, HuggingFace, etc.) read os.environ
    #    directly. pydantic-settings does NOT populate os.environ automatically.
    # -------------------------------------------------------------------------
    _env_exports = {
        "GROQ_API_KEY":          settings.groq_api_key,
        "HUGGINGFACE_API_KEY":   settings.huggingface_api_key,
        "NEO4J_URI":             settings.neo4j_uri,
        "QDRANT_API_KEY":        settings.qdrant_api_key,
    }
    for k, v in _env_exports.items():
        if v and not os.environ.get(k):
            os.environ[k] = v
            log.info(f"  Exported {k} to os.environ")
    # -----------------------------------------------------------------------
    # 1. Neo4j
    # -----------------------------------------------------------------------
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    try:
        neo4j.connect()
    except Exception as e:
        log.error(f"Neo4j connection failed: {e} — continuing in degraded mode")

    # -----------------------------------------------------------------------
    # 2. Qdrant
    # -----------------------------------------------------------------------
    qdrant = QdrantDB(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection_name,
        dim=settings.embedding_dim,
    )
    try:
        qdrant.connect()
    except Exception as e:
        log.error(f"Qdrant connection failed: {e} — continuing in degraded mode")

    # -----------------------------------------------------------------------
    # 3. In-memory DataStore (CSV files)
    # -----------------------------------------------------------------------
    data = DataStore(raw_dir=settings.raw_dir)
    try:
        data.load()
    except Exception as e:
        log.error(f"DataStore load failed: {e}")

    # -----------------------------------------------------------------------
    # 4. ML Service
    # -----------------------------------------------------------------------
    ml = MLService(
        models_dir=settings.models_dir,
        processed_dir=settings.processed_dir,
        evaluation_dir=settings.evaluation_dir,
    )
    try:
        ml.load()
    except Exception as e:
        log.error(f"MLService load failed: {e}")

    # -----------------------------------------------------------------------
    # 5. Embedding Service
    # -----------------------------------------------------------------------
    embedder = EmbedService(
        api_key=settings.huggingface_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )

    # -----------------------------------------------------------------------
    # 6. Graph Service & Vector Service
    # -----------------------------------------------------------------------
    graph_svc = GraphService(neo4j=neo4j)
    vector_svc = VectorService(qdrant=qdrant, embedder=embedder)
    try:
        vector_svc.setup()   # ensure collection + seed cases
    except Exception as e:
        log.error(f"VectorService setup failed: {e}")

    # -----------------------------------------------------------------------
    # Attach all services to app.state
    # -----------------------------------------------------------------------
    app.state.settings = settings
    app.state.neo4j    = neo4j
    app.state.qdrant   = qdrant
    app.state.data     = data
    app.state.ml       = ml
    app.state.embedder = embedder
    app.state.graph    = graph_svc
    app.state.vectors  = vector_svc

    log.info("=" * 60)
    log.info("  AbuseRing Sentinel — All services ready ✅")
    log.info("=" * 60)

    yield  # Application runs here

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------
    log.info("Shutting down AbuseRing Sentinel...")
    try:
        neo4j.close()
    except Exception:
        pass
    log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AbuseRing Sentinel API",
    description=(
        "AI-powered coordinated payment-abuse ring detection.\n\n"
        "Combines XGBoost ML, Neo4j graph intelligence, Qdrant case memory, "
        "and Groq LLM investigation agents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# #8: Register slowapi limiter on the app so @limiter.limit() decorators work
from .api.investigate import limiter as _investigate_limiter  # noqa: E402
app.state.limiter = _investigate_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
_PREFIX = "/api"

app.include_router(health.router,       prefix=_PREFIX)
app.include_router(dashboard.router,    prefix=_PREFIX)
app.include_router(clusters.router,     prefix=_PREFIX)
app.include_router(transactions.router, prefix=_PREFIX)
app.include_router(model.router,        prefix=_PREFIX)
app.include_router(rag.router,          prefix=_PREFIX)
app.include_router(investigate.router,  prefix=_PREFIX)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """#4: Return 503 instead of leaking raw 500s when Neo4j/Qdrant are degraded."""
    log.error(f"RuntimeError on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable", "reason": str(exc)},
    )


@app.get("/")
async def root():
    return {
        "name":    "AbuseRing Sentinel API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/health",
    }
