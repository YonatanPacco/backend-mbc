"""
Servicio de gestión del proceso DAQ — lanza, detiene y monitorea
el script daq_headless.py como subproceso.

Responsabilidades:
- Lifecycle del subproceso DAQ (start/stop)
- Polling del estado del HDF5 para reportar tasa de adquisición
- Interfaz pública para el router REST
"""

import os
import sys
import time
import subprocess
import threading

import h5py


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
_DAQ_SCRIPT        = os.path.join(os.path.dirname(__file__), "daq_headless.py")
_H5_ROOT_DIR       = os.path.join(os.path.dirname(__file__), "h5_data")
_PYTHON_EXECUTABLE = sys.executable

# Hardware G-Link 200 (coincide con procesamiento_nodos_zaranda.py)
MSCL_COM_PORT = "COM4"


def _current_hdf5_path() -> str | None:
    """Retorna el archivo HDF5 activo del bloque de 6h actual, o None si no existe."""
    from datetime import timezone, timedelta
    LIMA_TZ = timezone(timedelta(hours=-5))
    H5_ROTATE_HOURS = 6
    now = time.time()
    dt_local = __import__('datetime').datetime.fromtimestamp(now, tz=timezone.utc).astimezone(LIMA_TZ)
    day  = dt_local.strftime("%Y-%m-%d")
    hour = dt_local.hour
    h0   = (hour // H5_ROTATE_HOURS) * H5_ROTATE_HOURS
    h1   = min(23, h0 + H5_ROTATE_HOURS - 1)
    path = os.path.join(_H5_ROOT_DIR, day, f"data_{day}_{h0:02d}-{h1:02d}.h5")
    return path if os.path.exists(path) else None

# ---------------------------------------------------------------------------
# Estado interno (thread-safe)
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_process: subprocess.Popen | None = None
_start_time: float = 0.0
_mode: str = "mscl"
_daq_log_file = None

# Métricas de tasa HDF5
_monitor_lock = threading.Lock()
_last_sample_count: int = 0
_last_check_time: float = 0.0
_current_rate: float = 0.0
_total_samples: int = 0

# Hilo monitor
_monitor_thread: threading.Thread | None = None
_monitor_running = False


# ---------------------------------------------------------------------------
# Monitor de HDF5 — corre en background
# ---------------------------------------------------------------------------
def _monitor_loop() -> None:
    """Monitorea periódicamente el HDF5 para calcular tasa de muestras."""
    global _last_sample_count, _last_check_time, _current_rate, _total_samples

    while _monitor_running:
        time.sleep(2.0)

        if not _monitor_running:
            break

        try:
            hdf5_path = _current_hdf5_path()
            if hdf5_path is None:
                continue

            with h5py.File(hdf5_path, 'r', libver='latest', swmr=True) as f:
                if 'timestamp' not in f:
                    continue
                current_count = int(f['timestamp'].shape[0])

            now = time.time()

            with _monitor_lock:
                _total_samples = current_count

                if _last_check_time > 0:
                    dt = now - _last_check_time
                    dn = current_count - _last_sample_count
                    if dt > 0:
                        _current_rate = max(0.0, dn / dt)

                _last_sample_count = current_count
                _last_check_time = now

        except Exception:
            # HDF5 puede estar bloqueado por escritura — ignorar
            pass


def _start_monitor() -> None:
    """Inicia el hilo monitor si no está corriendo."""
    global _monitor_thread, _monitor_running, _last_sample_count, _last_check_time

    if _monitor_running:
        return

    _monitor_running = True
    _last_sample_count = 0
    _last_check_time = 0.0

    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        daemon=True,
        name="daq-monitor",
    )
    _monitor_thread.start()


def _stop_monitor() -> None:
    """Detiene el hilo monitor."""
    global _monitor_running
    _monitor_running = False


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------
def iniciar_daq(mode: str = "mscl") -> dict:
    """
    Lanza el proceso DAQ headless como subproceso.

    Returns:
        dict con status y mensaje
    """
    global _process, _start_time, _mode

    with _state_lock:
        if _process is not None and _process.poll() is None:
            return {
                "success": False,
                "message": "El DAQ ya está en ejecución",
                "pid": _process.pid,
            }

        _mode = mode

        cmd = [
            _PYTHON_EXECUTABLE,
            _DAQ_SCRIPT,
            "--mode", mode,
        ]

        if mode == "mscl":
            cmd += ["--port", MSCL_COM_PORT]

        try:
            global _daq_log_file
            # Log en backend/logs/ (separado del código fuente)
            log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "daq_output.log")
            log_path = os.path.normpath(log_path)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            _daq_log_file = open(log_path, "a", encoding="utf-8")
            
            _process = subprocess.Popen(
                cmd,
                stdout=_daq_log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            )
            _start_time = time.time()
            _start_monitor()

            print(f"[DAQ Manager] Proceso iniciado — PID: {_process.pid}, Modo: {mode}")

            return {
                "success": True,
                "message": f"DAQ iniciado en modo {mode}",
                "pid": _process.pid,
            }

        except Exception as error:
            return {
                "success": False,
                "message": f"Error al iniciar DAQ: {error}",
            }


def detener_daq() -> dict:
    """
    Detiene el proceso DAQ.

    Returns:
        dict con status y mensaje
    """
    global _process

    with _state_lock:
        if _process is None or _process.poll() is not None:
            _process = None
            _stop_monitor()
            return {
                "success": True,
                "message": "El DAQ no estaba en ejecución",
            }

        pid = _process.pid

        try:
            # En Windows usar CTRL_BREAK_EVENT, en Unix usar SIGTERM
            if os.name == 'nt':
                import signal
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                _process.terminate()

            # Esperar hasta 5 segundos
            try:
                _process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=3)

            _process = None
            
            if _daq_log_file:
                try:
                    _daq_log_file.close()
                except:
                    pass
                _daq_log_file = None

            _stop_monitor()

            print(f"[DAQ Manager] Proceso detenido — PID: {pid}")

            return {
                "success": True,
                "message": f"DAQ detenido (PID: {pid})",
                "pid": pid,
            }

        except Exception as error:
            _process = None
            _stop_monitor()
            return {
                "success": False,
                "message": f"Error al detener DAQ: {error}",
            }


def is_daq_running() -> bool:
    """Retorna True si el proceso DAQ está activo."""
    with _state_lock:
        return _process is not None and _process.poll() is None


def obtener_estado_daq() -> dict:
    """
    Retorna el estado actual del proceso DAQ y métricas HDF5.

    Returns:
        dict con estado completo
    """
    with _state_lock:
        is_running = _process is not None and _process.poll() is None
        pid = _process.pid if _process is not None and is_running else None

    uptime = time.time() - _start_time if is_running and _start_time > 0 else 0.0

    with _monitor_lock:
        rate = _current_rate
        total = _total_samples

    # Verificar archivo HDF5 activo
    hdf5_path    = _current_hdf5_path()
    hdf5_exists  = hdf5_path is not None
    hdf5_size_mb = 0.0
    if hdf5_exists:
        try:
            hdf5_size_mb = os.path.getsize(hdf5_path) / (1024 * 1024)
        except Exception:
            pass

    return {
        "running": is_running,
        "pid": pid,
        "mode": _mode if is_running else None,
        "uptime_seconds": round(uptime, 1),
        "total_samples": total,
        "sample_rate": round(rate, 1),
        "hdf5_exists": hdf5_exists,
        "hdf5_size_mb": round(hdf5_size_mb, 2),
    }
