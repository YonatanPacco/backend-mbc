"""
Esquemas Pydantic para validación de datos del dashboard.
Cada modelo define la estructura exacta de los datos que viajan por WebSocket.
"""

from pydantic import BaseModel, Field
from typing import List


class AceleracionData(BaseModel):
    """Datos de aceleración en los 3 ejes."""
    x: float = Field(..., description="Aceleración eje X en g")
    y: float = Field(..., description="Aceleración eje Y en g")
    z: float = Field(..., description="Aceleración eje Z en g")
    aceleracion_total: float = Field(..., description="Magnitud vectorial en g")
    unidad: str = "g"
    timestamp: float = Field(..., description="Epoch timestamp en segundos")


class FFTAxisData(BaseModel):
    """Datos FFT para un solo eje."""
    frecuencias: List[float] = Field(default_factory=list)
    amplitudes: List[float] = Field(default_factory=list)
    pico_frecuencia: float = Field(0.0, description="Frecuencia del pico dominante (Hz)")
    pico_amplitud: float = Field(0.0, description="Amplitud del pico dominante")


class FFTData(BaseModel):
    """Espectro FFT para los 3 ejes."""
    eje_x: FFTAxisData = Field(default_factory=FFTAxisData)
    eje_y: FFTAxisData = Field(default_factory=FFTAxisData)
    eje_z: FFTAxisData = Field(default_factory=FFTAxisData)
    sample_rate: float = Field(0.0, description="Frecuencia de muestreo usada (Hz)")
    ventana_muestras: int = Field(0, description="Número de muestras en la ventana FFT")
    timestamp: float = Field(0.0, description="Epoch timestamp del último dato")
