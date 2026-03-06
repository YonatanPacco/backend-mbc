"""
Servicio de métricas derivadas — calcula Stroke, RPM y Fase
a partir de los datos de aceleración.

Stroke se calcula por doble integración de la aceleración (pasa-alto).
RPM se estima del pico dominante de la FFT.
Fase se estima del desfase entre ejes.
"""

import math
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

from services.aceleracion_service import obtener_buffer_reciente, DEFAULT_NODE

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
G_TO_M_S2 = 9.80665
HP_CUTOFF = 0.5        # Hz — corte del filtro pasa-alto
HP_ORDER = 4
FFT_WINDOW = 512


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _butter_highpass(cutoff: float, fs: float, order: int = HP_ORDER):
    """Diseña filtro Butterworth pasa-alto."""
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        return None, None
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def _apply_highpass(data: np.ndarray, fs: float) -> np.ndarray:
    """Aplica filtro pasa-alto con fallback seguro."""
    if data.size < 16:
        return data
    result = _butter_highpass(HP_CUTOFF, fs)
    if result[0] is None:
        return data
    b, a = result
    try:
        return filtfilt(b, a, data)
    except Exception:
        return data


def _compute_stroke_mm(accel_g: np.ndarray, fs: float) -> float:
    """
    Calcula stroke (desplazamiento pico-a-pico) en mm
    por doble integración de la aceleración filtrada.
    """
    if accel_g.size < 16 or fs <= 0:
        return 0.0

    # Convertir g → m/s², filtrar
    accel_ms2 = accel_g * G_TO_M_S2
    filtered = _apply_highpass(accel_ms2, fs)

    dt = 1.0 / fs

    # Primera integración → velocidad (m/s)
    velocity = np.cumsum(filtered) * dt
    velocity -= np.mean(velocity)  # remover drift

    # Segunda integración → desplazamiento (m)
    displacement = np.cumsum(velocity) * dt
    displacement -= np.mean(displacement)  # remover drift

    # Stroke = pico-a-pico en mm
    stroke_mm = (np.max(displacement) - np.min(displacement)) * 1000.0
    return float(stroke_mm)


def _compute_rpm_and_phase(
    accel_x: np.ndarray,
    accel_y: np.ndarray,
    fs: float,
) -> tuple[float, float]:
    """
    Estima RPM del pico dominante de la FFT y fase entre X e Y.
    """
    if accel_x.size < 8 or fs <= 0:
        return 0.0, 0.0

    n = accel_x.size
    freqs = fftfreq(n, 1.0 / fs)

    # FFT del eje X
    fft_x = fft(accel_x * G_TO_M_S2)
    fft_y = fft(accel_y * G_TO_M_S2)

    # Solo frecuencias positivas (excluir DC)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    magnitudes_x = np.abs(fft_x[pos_mask])

    if pos_freqs.size == 0:
        return 0.0, 0.0

    # Pico dominante → RPM
    idx_peak = int(np.argmax(magnitudes_x))
    peak_freq = float(pos_freqs[idx_peak])
    rpm = peak_freq * 60.0  # Hz → RPM

    # Fase entre X e Y en el pico dominante
    phase_x = np.angle(fft_x[pos_mask][idx_peak])
    phase_y = np.angle(fft_y[pos_mask][idx_peak])
    phase_diff = math.degrees(phase_y - phase_x) % 360

    return round(rpm, 1), round(phase_diff, 2)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def obtener_metricas_resumen(node_id: int = DEFAULT_NODE) -> dict:
    """
    Calcula y retorna todas las métricas derivadas:
    Accel total, Stroke, RPM, Fase + tabla por eje.
    """
    buffer = obtener_buffer_reciente(FFT_WINDOW, node_id=node_id)

    x = buffer["x"]
    y = buffer["y"]
    z = buffer["z"]
    fs = buffer["sample_rate"]
    timestamps = buffer["timestamps"]

    timestamp = float(timestamps[-1]) if timestamps.size > 0 else 0.0

    # Si no hay datos suficientes, retornar ceros
    if x.size < 16:
        return {
            "accel_total": 0.0,
            "stroke_total": 0.0,
            "rpm": 0.0,
            "fase": 0.0,
            "ejes": {
                "x": {"accel_g": 0.0, "stroke_mm": 0.0},
                "y": {"accel_g": 0.0, "stroke_mm": 0.0},
                "z": {"accel_g": 0.0, "stroke_mm": 0.0},
            },
            "timestamp": timestamp,
        }

    # Aceleración RMS por eje (en g)
    rms_x = float(np.sqrt(np.mean(x**2)))
    rms_y = float(np.sqrt(np.mean(y**2)))
    rms_z = float(np.sqrt(np.mean(z**2)))
    accel_total = float(math.sqrt(rms_x**2 + rms_y**2 + rms_z**2))

    # Stroke por eje
    stroke_x = _compute_stroke_mm(x, fs)
    stroke_y = _compute_stroke_mm(y, fs)
    stroke_z = _compute_stroke_mm(z, fs)
    stroke_total = float(math.sqrt(stroke_x**2 + stroke_y**2 + stroke_z**2))

    # RPM y fase
    rpm, fase = _compute_rpm_and_phase(x, y, fs)

    return {
        "accel_total": round(accel_total, 2),
        "stroke_total": round(stroke_total, 2),
        "rpm": rpm,
        "fase": fase,
        "ejes": {
            "x": {"accel_g": round(rms_x, 2), "stroke_mm": round(stroke_x, 2)},
            "y": {"accel_g": round(rms_y, 2), "stroke_mm": round(stroke_y, 2)},
            "z": {"accel_g": round(rms_z, 2), "stroke_mm": round(stroke_z, 2)},
        },
        "timestamp": timestamp,
    }
