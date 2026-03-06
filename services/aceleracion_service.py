"""
Servicio de aceleración — 4 nodos, HDF5 rotativo.

Configuración basada en procesamiento_nodos_zaranda.py:
- node_id 1..4 (no las direcciones MSCL)
- Lee de h5_data/YYYY-MM-DD/data_YYYY-MM-DD_HH-HH.h5
- Buffer circular independiente por nodo
- Fallback a simulación cuando no hay DAQ activo
"""

import os
import time
import math
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Configuración (espeja procesamiento_nodos_zaranda.py)
# ---------------------------------------------------------------------------
SIMULATION_SAMPLE_RATE = 256.0
BUFFER_MAX_SIZE        = 2048
HDF5_READ_INTERVAL     = 0.2

H5_ROOT_DIR     = os.path.join(os.path.dirname(__file__), "h5_data")
H5_ROTATE_HOURS = 6
LIMA_TZ         = timezone(timedelta(hours=-5))

NODE_IDS    = [1, 2, 3, 4]   # node_id 1-4, no las direcciones
DEFAULT_NODE = 1


# ---------------------------------------------------------------------------
# Helpers HDF5 rotativo
# ---------------------------------------------------------------------------
def _block_start_hour(hour: int) -> int:
    return (hour // H5_ROTATE_HOURS) * H5_ROTATE_HOURS

def h5_current_path() -> str | None:
    """Retorna ruta del HDF5 activo en este bloque de 6h, None si no existe."""
    now      = time.time()
    dt_local = datetime.fromtimestamp(now, tz=timezone.utc).astimezone(LIMA_TZ)
    day      = dt_local.strftime("%Y-%m-%d")
    h0       = _block_start_hour(dt_local.hour)
    h1       = min(23, h0 + H5_ROTATE_HOURS - 1)
    path     = os.path.join(H5_ROOT_DIR, day, f"data_{day}_{h0:02d}-{h1:02d}.h5")
    return path if os.path.exists(path) else None


def _is_daq_running() -> bool:
    try:
        from services.daq_process_service import is_daq_running
        return is_daq_running()
    except Exception:
        return False


def _hdf5_active(max_age_seconds: float = 5.0) -> bool:
    path = h5_current_path()
    if path is None:
        return False
    try:
        return (time.time() - os.path.getmtime(path)) < max_age_seconds
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Buffers por nodo (thread-safe)
# ---------------------------------------------------------------------------
_buffer_lock = threading.Lock()

_node_buffers: dict = {}
_last_hdf5_timestamps: dict = {}
_last_ws_timestamps:   dict = {}

_acquiring         = False
_acquisition_thread = None
_hdf5_reader_active = False

_sim_t: float = 0.0


def _get_buffer(node_id: int) -> dict:
    if node_id not in _node_buffers:
        _node_buffers[node_id] = {
            'timestamps': deque(maxlen=BUFFER_MAX_SIZE),
            'x': deque(maxlen=BUFFER_MAX_SIZE),
            'y': deque(maxlen=BUFFER_MAX_SIZE),
            'z': deque(maxlen=BUFFER_MAX_SIZE),
        }
    return _node_buffers[node_id]


# ---------------------------------------------------------------------------
# Simulación (idéntica a procesamiento_nodos_zaranda.py)
# ---------------------------------------------------------------------------
def _generate_sim_samples(node_id: int, n: int) -> tuple:
    global _sim_t
    dt     = 1.0 / SIMULATION_SAMPLE_RATE
    now    = time.time()
    t_base = now - n * dt
    base   = 1.0 + 0.06 * (node_id - 1)

    ts_list, x_list, y_list, z_list = [], [], [], []
    for i in range(n):
        t  = _sim_t + i * dt
        ts = t_base + i * dt
        x  = base * (0.020 * math.sin(2 * math.pi * (3.0 + 0.2 * node_id) * t) + 0.005 * (2 * (i % 100) / 100 - 1))
        y  = base * (0.015 * math.sin(2 * math.pi * (5.0 + 0.2 * node_id) * t + 0.3) + 0.005 * (2 * (i % 73) / 73 - 1))
        z  = base * (0.010 * math.sin(2 * math.pi * (7.0 + 0.2 * node_id) * t + 1.0) + 0.005 * (2 * (i % 57) / 57 - 1))
        ts_list.append(ts); x_list.append(x); y_list.append(y); z_list.append(z)

    _sim_t += n * dt
    return ts_list, x_list, y_list, z_list


# ---------------------------------------------------------------------------
# Lector HDF5 rotativo
# ---------------------------------------------------------------------------
def _open_hdf5_reader(path: str):
    import shutil, tempfile
    try:
        return h5py.File(path, 'r', libver='latest', swmr=True)
    except Exception:
        pass
    try:
        return h5py.File(path, 'r', libver='latest')
    except Exception:
        pass
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix='.h5', prefix='daq_read_')
        os.close(fd)
        shutil.copy2(path, tmp)
        f = h5py.File(tmp, 'r', libver='latest')
        f._tmp_path = tmp
        return f
    except Exception:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return None


def _close_hdf5_reader(f):
    tmp = getattr(f, '_tmp_path', None)
    try:
        f.close()
    except Exception:
        pass
    if tmp:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _read_hdf5_latest() -> bool:
    """Lee muestras nuevas del HDF5 activo y distribuye a buffers por nodo."""
    global _hdf5_reader_active

    path = h5_current_path()
    if path is None:
        return False

    f = _open_hdf5_reader(path)
    if f is None:
        return False

    try:
        if 'timestamp' not in f or 'node' not in f:
            return False

        try:
            for ds in ('timestamp', 'node', 'ch1', 'ch2', 'ch3'):
                f[ds].id.refresh()
        except Exception:
            pass

        if f['timestamp'].shape[0] == 0:
            return False

        all_ts   = f['timestamp'][:]
        all_node = f['node'][:]
        all_ch1  = f['ch1'][:]
        all_ch2  = f['ch2'][:]
        all_ch3  = f['ch3'][:]

        got_any = False

        for node_id in NODE_IDS:
            last_ts  = _last_hdf5_timestamps.get(node_id, 0.0)
            node_mask = (all_node == node_id)
            node_ts   = all_ts[node_mask]

            if node_ts.size == 0:
                continue

            if last_ts > 0:
                new_mask = node_ts > last_ts
                if not np.any(new_mask):
                    continue
                first_new = int(np.argmax(new_mask))
            else:
                first_new = max(0, node_ts.size - BUFFER_MAX_SIZE)

            ts_new  = node_ts[first_new:]
            ch1_new = all_ch1[node_mask][first_new:]
            ch2_new = all_ch2[node_mask][first_new:]
            ch3_new = all_ch3[node_mask][first_new:]

            with _buffer_lock:
                buf = _get_buffer(node_id)
                for i in range(len(ts_new)):
                    buf['timestamps'].append(float(ts_new[i]))
                    buf['x'].append(float(ch1_new[i]))
                    buf['y'].append(float(ch2_new[i]))
                    buf['z'].append(float(ch3_new[i]))

            _last_hdf5_timestamps[node_id] = float(ts_new[-1])
            got_any = True

        return got_any

    except Exception as error:
        print(f"[ACQ Service] HDF5 read error: {error}")
        return False
    finally:
        _close_hdf5_reader(f)


# ---------------------------------------------------------------------------
# Hilo de adquisición
# ---------------------------------------------------------------------------
def _acquisition_loop() -> None:
    global _hdf5_reader_active
    sim_samples_per_tick = max(1, int(SIMULATION_SAMPLE_RATE * HDF5_READ_INTERVAL))

    while _acquiring:
        got_data = _read_hdf5_latest()

        if got_data:
            _hdf5_reader_active = True
            time.sleep(HDF5_READ_INTERVAL)
            continue

        _hdf5_reader_active = False

        if _hdf5_active():
            pass  # Escritor activo, sin datos nuevos aún
        elif _is_daq_running():
            with _buffer_lock:
                for node_id in NODE_IDS:
                    ts_list, x_list, y_list, z_list = _generate_sim_samples(node_id, sim_samples_per_tick)
                    buf = _get_buffer(node_id)
                    for ts, x, y, z in zip(ts_list, x_list, y_list, z_list):
                        buf['timestamps'].append(ts)
                        buf['x'].append(x)
                        buf['y'].append(y)
                        buf['z'].append(z)
        else:
            with _buffer_lock:
                for node_id in NODE_IDS:
                    buf = _get_buffer(node_id)
                    buf['timestamps'].clear(); buf['x'].clear()
                    buf['y'].clear(); buf['z'].clear()

        time.sleep(HDF5_READ_INTERVAL)


def iniciar_adquisicion() -> None:
    global _acquiring, _acquisition_thread
    if _acquiring:
        return
    _acquiring = True
    _acquisition_thread = threading.Thread(
        target=_acquisition_loop, daemon=True, name="acq-service"
    )
    _acquisition_thread.start()
    print("[ACQ Service] Adquisición iniciada (4 nodos, HDF5 rotativo)")


def detener_adquisicion() -> None:
    global _acquiring
    _acquiring = False
    print("[ACQ Service] Adquisición detenida")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def obtener_aceleracion(node_id: int = DEFAULT_NODE) -> dict:
    last_ts = _last_ws_timestamps.get(node_id, 0.0)

    with _buffer_lock:
        buf = _get_buffer(node_id)
        if not buf['timestamps']:
            return {
                "x": [], "y": [], "z": [], "timestamps": [],
                "aceleracion_total": 0.0, "unidad": "g",
                "fuente": "sin_datos", "node_id": node_id,
            }

        ts = np.array(buf['timestamps'])

        if last_ts > 0 and last_ts >= ts[0]:
            new_mask = ts > last_ts
            if not np.any(new_mask):
                return {
                    "x": [], "y": [], "z": [], "timestamps": [],
                    "aceleracion_total": 0.0, "unidad": "g",
                    "fuente": "esperando_daq", "node_id": node_id,
                }
            first_new = int(np.argmax(new_mask))
            if first_new == 0 and not new_mask[0]:
                return {
                    "x": [], "y": [], "z": [], "timestamps": [],
                    "aceleracion_total": 0.0, "unidad": "g",
                    "fuente": "esperando_daq", "node_id": node_id,
                }
        else:
            first_new = max(0, len(ts) - 50)

        ts_new = ts[first_new:]
        x_new  = np.array(buf['x'])[first_new:]
        y_new  = np.array(buf['y'])[first_new:]
        z_new  = np.array(buf['z'])[first_new:]

        ult_x    = float(x_new[-1]) if len(x_new) > 0 else 0.0
        ult_y    = float(y_new[-1]) if len(y_new) > 0 else 0.0
        ult_z    = float(z_new[-1]) if len(z_new) > 0 else 0.0
        magnitud = math.sqrt(ult_x**2 + ult_y**2 + ult_z**2)

        if len(ts_new) > 0:
            _last_ws_timestamps[node_id] = float(ts_new[-1])

    return {
        "x": [float(v) for v in x_new.tolist()],
        "y": [float(v) for v in y_new.tolist()],
        "z": [float(v) for v in z_new.tolist()],
        "timestamps": [float(v) for v in ts_new.tolist()],
        "aceleracion_total": round(magnitud, 6),
        "unidad": "g",
        "fuente": "hdf5" if _hdf5_reader_active else "simulacion",
        "node_id": node_id,
    }


def obtener_buffer_reciente(num_muestras: int = 512, node_id: int = DEFAULT_NODE) -> dict:
    with _buffer_lock:
        buf = _get_buffer(node_id)
        n   = min(num_muestras, len(buf['timestamps']))
        if n == 0:
            return {
                "timestamps": np.array([]), "x": np.array([]),
                "y": np.array([]),          "z": np.array([]),
                "sample_rate": SIMULATION_SAMPLE_RATE, "node_id": node_id,
            }

        ts = np.array(list(buf['timestamps'])[-n:], dtype=np.float64)
        x  = np.array(list(buf['x'])[-n:],          dtype=np.float64)
        y  = np.array(list(buf['y'])[-n:],          dtype=np.float64)
        z  = np.array(list(buf['z'])[-n:],          dtype=np.float64)

    sample_rate = SIMULATION_SAMPLE_RATE
    if ts.size >= 2:
        dt_median = np.median(np.diff(ts))
        if dt_median > 0:
            sample_rate = 1.0 / dt_median

    return {
        "timestamps": ts, "x": x, "y": y, "z": z,
        "sample_rate": sample_rate, "node_id": node_id,
    }
