"""
Router REST para datos históricos del HDF5 rotativo.
Endpoint: GET /api/historico?node=1
Devuelve datos del bloque HDF5 activo filtrados por node_id (1-4),
downsampled a 2000 pts máx.
"""

import numpy as np
from fastapi import APIRouter, Query

router = APIRouter()

MAX_POINTS = 2000


@router.get("/historico")
async def get_historico(node: int = Query(default=1, ge=1, le=4)) -> dict:
    """Retorna datos históricos del HDF5 rotativo activo, filtrados por nodo."""
    empty = {"timestamps": [], "x": [], "y": [], "z": [], "total": 0, "node": node}
    try:
        import h5py
        from services.aceleracion_service import h5_current_path

        path = h5_current_path()
        if path is None:
            return empty

        with h5py.File(path, 'r', swmr=True) as f:
            if 'timestamp' not in f or f['timestamp'].shape[0] == 0:
                return empty

            node_col = f['node'][:]
            mask = node_col == node
            if not mask.any():
                return empty

            ts = f['timestamp'][:][mask]
            x  = f['ch1'][:][mask]
            y  = f['ch2'][:][mask]
            z  = f['ch3'][:][mask]

        total = len(ts)
        step  = max(1, total // MAX_POINTS)

        return {
            "timestamps": [float(v) for v in ts[::step].tolist()],
            "x":  [float(v) for v in x[::step].tolist()],
            "y":  [float(v) for v in y[::step].tolist()],
            "z":  [float(v) for v in z[::step].tolist()],
            "total": total,
            "node": node,
        }

    except Exception as e:
        print(f"[Historico] Error: {e}")
        return empty
