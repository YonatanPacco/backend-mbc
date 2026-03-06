"""
Servicio FFT — calcula el espectro de frecuencias a partir del buffer
de aceleración.

Reutiliza la lógica de calculate_fft() de aceleracion_to_influxdb.py
adaptada para retornar datos por eje y picos dominantes.
"""

import numpy as np
from scipy.fft import fft, fftfreq

from services.aceleracion_service import obtener_buffer_reciente, DEFAULT_NODE


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
FFT_WINDOW_SIZE = 512   # Muestras para FFT (potencia de 2 para eficiencia)
MAX_FREQ_POINTS = 256   # Limitar puntos enviados al frontend


# ---------------------------------------------------------------------------
# Cálculo FFT para un eje (adaptado de aceleracion_to_influxdb.py)
# ---------------------------------------------------------------------------
def _calcular_fft_eje(data: np.ndarray, sample_rate: float) -> dict:
    """
    Calcula FFT de una señal y retorna frecuencias positivas + magnitudes.
    Replica la lógica de calculate_fft() de aceleracion_to_influxdb.py.
    """
    if data.size < 4:
        return {
            "frecuencias": [],
            "amplitudes": [],
            "pico_frecuencia": 0.0,
            "pico_amplitud": 0.0,
        }

    n_samples = data.size
    yf = fft(data)
    xf = fftfreq(n_samples, 1.0 / sample_rate)

    # Solo frecuencias positivas
    positive_mask = xf >= 0
    freqs = xf[positive_mask]
    magnitudes = (2.0 / n_samples) * np.abs(yf[positive_mask])

    # Excluir componente DC (índice 0)
    if freqs.size > 1:
        freqs = freqs[1:]
        magnitudes = magnitudes[1:]

    # Limitar cantidad de puntos enviados
    if freqs.size > MAX_FREQ_POINTS:
        step = max(1, freqs.size // MAX_FREQ_POINTS)
        freqs = freqs[::step]
        magnitudes = magnitudes[::step]

    # Pico dominante
    pico_frecuencia = 0.0
    pico_amplitud = 0.0
    if magnitudes.size > 0:
        idx_max = int(np.argmax(magnitudes))
        pico_frecuencia = float(freqs[idx_max])
        pico_amplitud = float(magnitudes[idx_max])

    return {
        "frecuencias": [float(f) for f in freqs.tolist()],
        "amplitudes": [float(m) for m in magnitudes.tolist()],
        "pico_frecuencia": pico_frecuencia,
        "pico_amplitud": pico_amplitud,
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def calcular_fft_completo(node_id: int = DEFAULT_NODE) -> dict:
    """
    Calcula FFT para los 3 ejes usando el buffer de aceleración más reciente.
    Retorna un diccionario listo para enviar al frontend via WebSocket.
    """
    buffer = obtener_buffer_reciente(FFT_WINDOW_SIZE, node_id=node_id)

    datos_x = buffer["x"]
    datos_y = buffer["y"]
    datos_z = buffer["z"]
    sample_rate = buffer["sample_rate"]
    timestamps = buffer["timestamps"]

    fft_x = _calcular_fft_eje(datos_x, sample_rate)
    fft_y = _calcular_fft_eje(datos_y, sample_rate)
    fft_z = _calcular_fft_eje(datos_z, sample_rate)

    timestamp = float(timestamps[-1]) if timestamps.size > 0 else 0.0

    return {
        "eje_x": fft_x,
        "eje_y": fft_y,
        "eje_z": fft_z,
        "sample_rate": round(sample_rate, 2),
        "ventana_muestras": int(datos_x.size),
        "timestamp": timestamp,
    }
