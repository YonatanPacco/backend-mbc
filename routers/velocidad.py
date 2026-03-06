"""
Router WebSocket para velocidad en tiempo real.
Endpoint: /ws/velocidad?node=1   (node = 1..4)
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.velocidad_service import obtener_velocidad
from services.aceleracion_service import DEFAULT_NODE

router = APIRouter()

ENVIO_INTERVALO = 1.0


@router.websocket("/velocidad")
async def websocket_velocidad(
    websocket: WebSocket,
    node: int = Query(default=DEFAULT_NODE),
) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[WS Velocidad] Cliente conectado: {client_host} | Nodo: {node}")

    try:
        while True:
            data = obtener_velocidad(node_id=node)
            await websocket.send_json(data)
            await asyncio.sleep(ENVIO_INTERVALO)
    except WebSocketDisconnect:
        print(f"[WS Velocidad] Cliente desconectado: {client_host}")
    except Exception as error:
        print(f"[WS Velocidad] Error: {error}")
