"""
Router REST para control del proceso DAQ.

Endpoints:
    POST /api/daq/start   → Inicia el proceso DAQ
    POST /api/daq/stop    → Detiene el proceso DAQ
    GET  /api/daq/status   → Retorna estado actual del DAQ
"""

from fastapi import APIRouter
from pydantic import BaseModel

from services.daq_process_service import (
    iniciar_daq,
    detener_daq,
    obtener_estado_daq,
)

router = APIRouter()


class DaqStartRequest(BaseModel):
    """Cuerpo de la petición para iniciar el DAQ."""
    mode: str = "mscl"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/daq/start")
async def start_daq(request: DaqStartRequest) -> dict:
    """Inicia el proceso de adquisición DAQ."""
    result = iniciar_daq(mode=request.mode)
    return result


@router.post("/daq/stop")
async def stop_daq() -> dict:
    """Detiene el proceso de adquisición DAQ."""
    result = detener_daq()
    return result


@router.get("/daq/status")
async def get_daq_status() -> dict:
    """Retorna el estado actual del proceso DAQ y métricas HDF5."""
    return obtener_estado_daq()
