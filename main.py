"""
Punto de entrada del backend FastAPI — Dashboard de Confiabilidad de Activos.

Registra los routers WebSocket, configura CORS y arranca el hilo
de adquisición de datos al iniciar el servidor.

Uso:
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Carga .env antes de cualquier import que lea os.getenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import aceleracion, fft, metricas, daq_control, velocidad, historico
from routers import auth
from services.aceleracion_service import iniciar_adquisicion, detener_adquisicion
from services.daq_process_service import detener_daq


# ---------------------------------------------------------------------------
# Lifecycle — arrancar/detener adquisición con el servidor
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicia la adquisición al arrancar y la detiene al cerrar."""
    iniciar_adquisicion()
    yield
    detener_adquisicion()
    detener_daq()  # Asegurar que el subproceso DAQ se detenga al cerrar


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dashboard API — Confiabilidad de Activos",
    description="Backend WebSocket para análisis de señales en tiempo real",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permite al frontend Astro conectarse
ALLOWED_ORIGINS = [
    "http://localhost:4321",       # Astro dev server
    "http://localhost:4322",       # Astro dev server (fallback port)
    "http://localhost:3000",       # Fallback
    "http://127.0.0.1:4321",
    "http://127.0.0.1:4322",
    "http://127.0.0.1:3000",
    "http://192.168.1.4:4321",
    "https://mbcpredictive.com",   # Producción
    "https://www.mbcpredictive.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Registrar routers WebSocket
# ---------------------------------------------------------------------------
app.include_router(auth.router,        prefix="/api", tags=["Auth"])
app.include_router(aceleracion.router, prefix="/ws",  tags=["WebSocket"])
app.include_router(fft.router,         prefix="/ws",  tags=["WebSocket"])
app.include_router(metricas.router,    prefix="/ws",  tags=["WebSocket"])
app.include_router(velocidad.router,   prefix="/ws",  tags=["WebSocket"])
app.include_router(daq_control.router, prefix="/api", tags=["DAQ Control"])
app.include_router(historico.router,   prefix="/api", tags=["Histórico"])


# ---------------------------------------------------------------------------
# Endpoints REST
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint de verificación de salud del servidor."""
    return {"status": "ok", "service": "dashboard-api"}
