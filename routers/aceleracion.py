"""
Router WebSocket para datos de aceleración en tiempo real.
Endpoint: /ws/aceleracion?node=1   (node = 1..4)
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.aceleracion_service import obtener_aceleracion, DEFAULT_NODE  # DEFAULT_NODE = 1

router = APIRouter()

ENVIO_HZ = 20
ENVIO_INTERVALO = 1.0 / ENVIO_HZ


@router.websocket("/aceleracion")
async def websocket_aceleracion(
    websocket: WebSocket,
    node: int = Query(default=DEFAULT_NODE),
) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[WS Aceleración] Cliente conectado: {client_host} | Nodo: {node}")

    try:
        while True:
            data = obtener_aceleracion(node_id=node)
            await websocket.send_json(data)
            await asyncio.sleep(ENVIO_INTERVALO)
    except WebSocketDisconnect:
        print(f"[WS Aceleración] Cliente desconectado: {client_host}")
    except Exception as error:
        print(f"[WS Aceleración] Error: {error}")
