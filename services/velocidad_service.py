"""
Servicio de velocidad — integra aceleración a velocidad.

Replica la lógica de VelocityWindow en Daqaceleracionglink200_1.py:
  1. High-pass filter 0.5 Hz orden 4 (elimina DC / gravedad)
  2. Integración trapezoidal → mm/s
  3. Substracción de media (corrección de drift)
  4. Rolling RMS (ventana 1s)
  5. Historial de VRMS (tendencia últimos 5 minutos)
"""

import time
import threading
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

from services.aceleracion_service import obtener_buffer_reciente, SIMULATION_SAMPLE_RATE, DEFAULT_NODE

# ---------------------------------------------------------------------------
# Configuración (espeja Daqaceleracionglink200_1.py)
# ---------------------------------------------------------------------------
HP_CUTOFF            = 0.5      # Hz
HP_ORDER             = 4
G_TO_M_S2            = 9.80665
RMS_WINDOW_SEC       = 1.0
ALARM_THRESHOLD_MM_S = 5.0
TREND_MAX_POINTS     = 300      # ~5 min a 1 punto/s
MAX_POINTS_OUT       = 512      # puntos máx enviados al frontend

# ---------------------------------------------------------------------------
# Historial VRMS (tendencia) — por nodo
# ---------------------------------------------------------------------------
_trend_lock = threading.Lock()
_trend_data: dict = {}  # { node_id: { ts, rx, ry, rz } }


def _get_trend(node_id: int) -> dict:
    if node_id not in _trend_data:
        _trend_data[node_id] = {
            'ts': deque(maxlen=TREND_MAX_POINTS),
            'rx': deque(maxlen=TREND_MAX_POINTS),
            'ry': deque(maxlen=TREND_MAX_POINTS),
            'rz': deque(maxlen=TREND_MAX_POINTS),
        }
    return _trend_data[node_id]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _apply_highpass(data: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.size < max(16, HP_ORDER * 4):
        return data
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        return data
    try:
        nc = cutoff / nyq
        b, a = butter(HP_ORDER, nc, btype='high', analog=False)
        if np.isnan(data).any():
            data = np.nan_to_num(data)
        return filtfilt(b, a, data)
    except Exception:
        return data


def _rolling_rms(x: np.ndarray, nwin: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if nwin <= 1 or x.size == 0:
        return np.sqrt(x ** 2)
    kernel = np.ones(nwin, dtype=float) / nwin
    y = np.convolve(x ** 2, kernel, mode='same')
    return np.sqrt(np.maximum(y, 0.0))


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def obtener_velocidad(node_id: int = DEFAULT_NODE) -> dict:
    """
    Calcula velocidad instantánea y RMS desde el buffer de aceleración.
    """
    buf = obtener_buffer_reciente(2048, node_id=node_id)

    ts   = buf["timestamps"]
    ax_g = buf["x"]
    ay_g = buf["y"]
    az_g = buf["z"]
    fs   = buf["sample_rate"] or SIMULATION_SAMPLE_RATE

    if ts.size < 8:
        return _empty_response()

    # 1. Convertir a m/s²
    ax = ax_g * G_TO_M_S2
    ay = ay_g * G_TO_M_S2
    az = az_g * G_TO_M_S2

    # 2. High-pass filter (elimina gravedad y DC)
    fax = _apply_highpass(ax, HP_CUTOFF, fs)
    fay = _apply_highpass(ay, HP_CUTOFF, fs)
    faz = _apply_highpass(az, HP_CUTOFF, fs)

    # 3. Integración trapezoidal → mm/s
    try:
        vx = cumulative_trapezoid(fax, ts, initial=0.0) * 1000.0
        vy = cumulative_trapezoid(fay, ts, initial=0.0) * 1000.0
        vz = cumulative_trapezoid(faz, ts, initial=0.0) * 1000.0
    except Exception:
        dt = 1.0 / fs
        vx = np.cumsum(fax) * dt * 1000.0
        vy = np.cumsum(fay) * dt * 1000.0
        vz = np.cumsum(faz) * dt * 1000.0

    # 4. Substracción de media (drift)
    try:
        vx -= np.nanmean(vx)
        vy -= np.nanmean(vy)
        vz -= np.nanmean(vz)
    except Exception:
        pass

    # 5. Rolling RMS (ventana 1s)
    nwin = max(1, int(RMS_WINDOW_SEC * fs))
    rx = _rolling_rms(vx, nwin)
    ry = _rolling_rms(vy, nwin)
    rz = _rolling_rms(vz, nwin)

    # Valores actuales
    xv = float(rx[-1]) if rx.size else 0.0
    yv = float(ry[-1]) if ry.size else 0.0
    zv = float(rz[-1]) if rz.size else 0.0
    alarm = (xv >= ALARM_THRESHOLD_MM_S) or (yv >= ALARM_THRESHOLD_MM_S) or (zv >= ALARM_THRESHOLD_MM_S)

    # Actualizar tendencia (un punto por llamada, ~1/s)
    now = float(ts[-1]) if ts.size else time.time()
    with _trend_lock:
        trend = _get_trend(node_id)
        trend['ts'].append(now)
        trend['rx'].append(xv)
        trend['ry'].append(yv)
        trend['rz'].append(zv)

        t_trend  = list(trend['ts'])
        rx_trend = list(trend['rx'])
        ry_trend = list(trend['ry'])
        rz_trend = list(trend['rz'])

    # Downsample para limitar payload
    step = max(1, ts.size // MAX_POINTS_OUT)
    ts_out = ts[::step]
    vx_out = vx[::step]
    vy_out = vy[::step]
    vz_out = vz[::step]
    rx_out = rx[::step]
    ry_out = ry[::step]
    rz_out = rz[::step]

    return {
        "timestamps": [float(v) for v in ts_out.tolist()],
        "vx": [float(v) for v in vx_out.tolist()],
        "vy": [float(v) for v in vy_out.tolist()],
        "vz": [float(v) for v in vz_out.tolist()],
        "rx": [float(v) for v in rx_out.tolist()],
        "ry": [float(v) for v in ry_out.tolist()],
        "rz": [float(v) for v in rz_out.tolist()],
        "vrms_actual": {
            "x": round(xv, 4),
            "y": round(yv, 4),
            "z": round(zv, 4),
        },
        "alarm": alarm,
        "alarm_threshold": ALARM_THRESHOLD_MM_S,
        "trend": {
            "timestamps": t_trend,
            "rx": rx_trend,
            "ry": ry_trend,
            "rz": rz_trend,
        },
    }


def _empty_response() -> dict:
    return {
        "timestamps": [], "vx": [], "vy": [], "vz": [],
        "rx": [], "ry": [], "rz": [],
        "vrms_actual": {"x": 0.0, "y": 0.0, "z": 0.0},
        "alarm": False,
        "alarm_threshold": ALARM_THRESHOLD_MM_S,
        "trend": {"timestamps": [], "rx": [], "ry": [], "rz": []},
    }
