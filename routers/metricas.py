"""
Router WebSocket para métricas resumen en tiempo real.
Endpoint: /ws/metricas?node=1   (node = 1..4)
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.metricas_service import obtener_metricas_resumen
from services.aceleracion_service import DEFAULT_NODE

router = APIRouter()

METRICAS_INTERVALO = 1.0


@router.websocket("/metricas")
async def websocket_metricas(
    websocket: WebSocket,
    node: int = Query(default=DEFAULT_NODE),
) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[WS Métricas] Cliente conectado: {client_host} | Nodo: {node}")

    try:
        while True:
            data = obtener_metricas_resumen(node_id=node)
            await websocket.send_json(data)
            await asyncio.sleep(METRICAS_INTERVALO)
    except WebSocketDisconnect:
        print(f"[WS Métricas] Cliente desconectado: {client_host}")
    except Exception as error:
        print(f"[WS Métricas] Error: {error}")
