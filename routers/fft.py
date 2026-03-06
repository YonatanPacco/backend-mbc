"""
Router WebSocket para espectro FFT en tiempo real.
Endpoint: /ws/fft?node=1   (node = 1..4)
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.fft_service import calcular_fft_completo
from services.aceleracion_service import DEFAULT_NODE

router = APIRouter()

FFT_INTERVALO = 0.5


@router.websocket("/fft")
async def websocket_fft(
    websocket: WebSocket,
    node: int = Query(default=DEFAULT_NODE),
) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[WS FFT] Cliente conectado: {client_host} | Nodo: {node}")

    try:
        while True:
            data = calcular_fft_completo(node_id=node)
            await websocket.send_json(data)
            await asyncio.sleep(FFT_INTERVALO)
    except WebSocketDisconnect:
        print(f"[WS FFT] Cliente desconectado: {client_host}")
    except Exception as error:
        print(f"[WS FFT] Error: {error}")
