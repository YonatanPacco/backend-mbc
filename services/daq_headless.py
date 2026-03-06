#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daq_headless.py — Versión headless del DAQ para 4 nodos.

Configuración basada en procesamiento_nodos_zaranda.py:
- 4 nodos G-Link 200 (NODE_ADDRESSES = [33190, 33191, 33192, 33193])
- node_id 1-4 (no las direcciones reales)
- HDF5 rotativo cada 6h: h5_data/YYYY-MM-DD/data_YYYY-MM-DD_HH-HH.h5
- Schema: timestamp, datetime, node (1-4), ch1, ch2, ch3

Uso:
    python daq_headless.py --mode simulation
    python daq_headless.py --mode mscl --port COM4
"""

import sys
import os
import time
import math
import signal
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import h5py

# =====================================================================
# CONFIGURACIÓN (espeja procesamiento_nodos_zaranda.py)
# =====================================================================
COM_PORT       = "COM4"
NODE_ADDRESSES = [33190, 33191, 33192, 33193]
NODE_COUNT     = 4

H5_ROOT_DIR      = os.path.join(os.path.dirname(__file__), "h5_data")
H5_ROTATE_HOURS  = 6
LIMA_TZ          = timezone(timedelta(hours=-5))

SIM_FS = 256.0
SIM_DT = 1.0 / SIM_FS

WRITE_FLUSH_INTERVAL = 0.5
FILE_LOCK    = threading.Lock()
PENDING_LOCK = threading.Lock()
PENDING_MAX  = 2_000_000
PENDING_WRITES: deque = deque(maxlen=PENDING_MAX)

_running       = True
_sample_count  = 0
_start_time    = 0.0


# =====================================================================
# Señal de parada limpia
# =====================================================================
def _handle_signal(signum, frame):
    global _running
    print(f"\n[DAQ Headless] Señal recibida ({signum}), deteniendo...")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# =====================================================================
# HDF5 rotativo — helpers (idéntico a procesamiento_nodos_zaranda.py)
# =====================================================================
def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _block_start_hour(hour: int) -> int:
    return (hour // H5_ROTATE_HOURS) * H5_ROTATE_HOURS

def h5_path_for_ts(ts_epoch: float) -> str:
    dt_local = datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).astimezone(LIMA_TZ)
    day  = dt_local.strftime("%Y-%m-%d")
    hour = dt_local.hour
    h0   = _block_start_hour(hour)
    h1   = min(23, h0 + H5_ROTATE_HOURS - 1)
    folder = os.path.join(H5_ROOT_DIR, day)
    _ensure_dir(folder)
    fname = f"data_{day}_{h0:02d}-{h1:02d}.h5"
    return os.path.join(folder, fname)

_ensure_dir(H5_ROOT_DIR)


# =====================================================================
# HDF5 helpers
# =====================================================================
def create_empty_hdf5(path: str):
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    chunk = (8192,)
    with h5py.File(path, 'w', libver='latest') as f:
        f.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype='f8', chunks=chunk)
        f.create_dataset('datetime',  shape=(0,), maxshape=(None,),
                         dtype=h5py.string_dtype('utf-8'), chunks=chunk)
        f.create_dataset('node',      shape=(0,), maxshape=(None,), dtype='u2', chunks=chunk)
        f.create_dataset('ch1',       shape=(0,), maxshape=(None,), dtype='f4', chunks=chunk)
        f.create_dataset('ch2',       shape=(0,), maxshape=(None,), dtype='f4', chunks=chunk)
        f.create_dataset('ch3',       shape=(0,), maxshape=(None,), dtype='f4', chunks=chunk)
        f.flush()

def initialize_hdf5_for_path(path: str):
    if os.path.exists(path):
        try:
            with h5py.File(path, 'r', libver='latest') as f:
                for k in ('timestamp', 'datetime', 'node', 'ch1', 'ch2', 'ch3'):
                    if k not in f:
                        raise RuntimeError(f"dataset faltante: {k}")
            return
        except Exception:
            try:
                os.replace(path, path + ".bak")
            except Exception:
                pass
    create_empty_hdf5(path)

def append_to_pending(node_id: int, ts: float, dt_string: str,
                      ch1: float, ch2: float, ch3: float):
    with PENDING_LOCK:
        PENDING_WRITES.append((float(ts), str(dt_string), int(node_id),
                               float(ch1), float(ch2), float(ch3)))


def _write_batch_to_file(path: str, batch_items):
    initialize_hdf5_for_path(path)
    chunk = (8192,)
    with FILE_LOCK:
        with h5py.File(path, 'a', libver='latest') as f:
            for k in ('timestamp', 'datetime', 'node', 'ch1', 'ch2', 'ch3'):
                if k not in f:
                    if k == 'datetime':
                        dtype = h5py.string_dtype('utf-8')
                    elif k == 'timestamp':
                        dtype = 'f8'
                    elif k == 'node':
                        dtype = 'u2'
                    else:
                        dtype = 'f4'
                    f.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dtype, chunks=chunk)
            try:
                f.swmr_mode = True
            except Exception:
                pass

            cur   = f['timestamp'].shape[0]
            n_add = len(batch_items)
            if n_add == 0:
                return
            new = cur + n_add

            for k in ('timestamp', 'datetime', 'node', 'ch1', 'ch2', 'ch3'):
                f[k].resize((new,))

            ts_arr = np.array([b[0] for b in batch_items], dtype='f8')
            dt_arr = [b[1] for b in batch_items]
            nd_arr = np.array([b[2] for b in batch_items], dtype='u2')
            c1_arr = np.array([b[3] for b in batch_items], dtype='f4')
            c2_arr = np.array([b[4] for b in batch_items], dtype='f4')
            c3_arr = np.array([b[5] for b in batch_items], dtype='f4')

            sl = slice(cur, new)
            f['timestamp'][sl] = ts_arr
            try:
                f['datetime'][sl] = [str(s) for s in dt_arr]
            except Exception:
                for i, v in enumerate(dt_arr):
                    f['datetime'][cur + i] = str(v)
            f['node'][sl] = nd_arr
            f['ch1'][sl]  = c1_arr
            f['ch2'][sl]  = c2_arr
            f['ch3'][sl]  = c3_arr
            f.flush()


def flush_pending_loop():
    print("[DAQ Headless] Hilo de escritura HDF5 rotativo iniciado.")
    while _running:
        time.sleep(WRITE_FLUSH_INTERVAL)

        with PENDING_LOCK:
            if not PENDING_WRITES:
                continue
            batch = list(PENDING_WRITES)
            PENDING_WRITES.clear()

        try:
            # Ordenar por (ts, node) y deduplicar
            ts_arr = np.array([b[0] for b in batch], dtype='f8')
            nd_arr = np.array([b[2] for b in batch], dtype='u2')
            idx    = np.lexsort((nd_arr, ts_arr))
            batch  = [batch[i] for i in idx]

            dedup, last_key = [], None
            for b in batch:
                key = (b[0], b[2])
                if key != last_key:
                    dedup.append(b)
                    last_key = key
            batch = dedup
            if not batch:
                continue

        except Exception as e:
            print(f"[DAQ Headless] sort/dedup error: {e}")
            continue

        # Agrupar por archivo de destino (bloque 6h)
        groups: dict = {}
        for it in batch:
            path = h5_path_for_ts(it[0])
            groups.setdefault(path, []).append(it)

        try:
            for path, items in groups.items():
                _write_batch_to_file(path, items)
        except Exception as e:
            print(f"[DAQ Headless] Write error: {e}")
            with PENDING_LOCK:
                for it in reversed(batch):
                    PENDING_WRITES.appendleft(it)
            time.sleep(0.2)

    print("[DAQ Headless] Hilo de escritura HDF5 finalizado.")


# =====================================================================
# Simulación — idéntica a procesamiento_nodos_zaranda.py
# =====================================================================
def simulate_sample(node_id: int, t: float):
    base = 1.0 + 0.06 * (node_id - 1)
    ax = base * (0.020 * math.sin(2 * math.pi * (3.0 + 0.2 * node_id) * t) + 0.005 * np.random.randn())
    ay = base * (0.015 * math.sin(2 * math.pi * (5.0 + 0.2 * node_id) * t + 0.3) + 0.005 * np.random.randn())
    az = base * (0.010 * math.sin(2 * math.pi * (7.0 + 0.2 * node_id) * t + 1.0) + 0.005 * np.random.randn())
    return float(ax), float(ay), float(az)


# =====================================================================
# Loops de adquisición
# =====================================================================
def run_simulation():
    global _sample_count, _start_time
    sim_t      = 0.0
    _start_time = time.time()

    print(f"[DAQ Headless] Simulación 4 nodos a {SIM_FS} Hz")
    print(f"[DAQ Headless] HDF5 dir: {H5_ROOT_DIR}")

    while _running:
        ts_epoch  = time.time()
        dt_local  = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).astimezone(LIMA_TZ)
        dt_string = dt_local.strftime("%Y-%m-%d %H:%M:%S%z")

        for node_id in range(1, NODE_COUNT + 1):
            ax, ay, az = simulate_sample(node_id, sim_t)
            append_to_pending(node_id, ts_epoch, dt_string, ax, ay, az)

        sim_t += SIM_DT
        _sample_count += NODE_COUNT

        if _sample_count % (500 * NODE_COUNT) == 0:
            elapsed = time.time() - _start_time
            rate    = _sample_count / elapsed if elapsed > 0 else 0
            print(f"[DAQ Headless] Muestras totales: {_sample_count} | Tasa: {rate:.1f}/s")

        time.sleep(SIM_DT)


def run_mscl(port: str = COM_PORT):
    global _sample_count, _start_time

    try:
        from python_mscl import mscl
    except ImportError:
        print("[DAQ Headless] ERROR: python_mscl no disponible. Use --mode simulation")
        return

    _start_time  = time.time()
    addr_to_node: dict = {}
    for i in range(min(NODE_COUNT, len(NODE_ADDRESSES))):
        addr_to_node[int(NODE_ADDRESSES[i])] = i + 1

    def assign_address(addr: int):
        if addr in addr_to_node:
            return addr_to_node[addr]
        if len(addr_to_node) >= NODE_COUNT:
            return None
        used = set(addr_to_node.values())
        for nid in range(1, NODE_COUNT + 1):
            if nid not in used:
                addr_to_node[addr] = nid
                print(f"[DAQ Headless] Auto-asignado address {addr} -> Nodo {nid}")
                return nid
        return None

    try:
        conn         = mscl.Connection.Serial(port)
        base_station = mscl.BaseStation(conn)
        print(f"[DAQ Headless] MSCL conectado en {port} | Nodos: {NODE_ADDRESSES}")
    except Exception as error:
        print(f"[DAQ Headless] ERROR MSCL: {error}")
        return

    while _running:
        try:
            sweeps = base_station.getData(200)
        except Exception as error:
            print(f"[DAQ Headless] getData error: {error}")
            sweeps = []

        for sweep in sweeps:
            if not _running:
                break
            try:
                addr    = int(sweep.nodeAddress())
                node_id = assign_address(addr)
                if node_id is None:
                    continue

                try:
                    ts_epoch = float(sweep.timestamp().nanoseconds()) / 1e9
                except Exception:
                    ts_epoch = time.time()

                dt_local  = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).astimezone(LIMA_TZ)
                dt_string = dt_local.strftime("%Y-%m-%d %H:%M:%S%z")

                ax = ay = az = None
                for dp in sweep.data():
                    name = dp.channelName().lower()
                    try:
                        v = dp.as_float()
                    except Exception:
                        try:
                            v = float(dp.as_string())
                        except Exception:
                            v = 0.0

                    if ("accel" in name and "x" in name) or name == "x" or "ch1" in name:
                        ax = v
                    elif ("accel" in name and "y" in name) or name == "y" or "ch2" in name:
                        ay = v
                    elif ("accel" in name and "z" in name) or name == "z" or "ch3" in name:
                        az = v

                if ax is not None and ay is not None and az is not None:
                    append_to_pending(node_id, ts_epoch, dt_string, ax, ay, az)
                    _sample_count += 1

            except Exception as error:
                print(f"[DAQ Headless] Parse sweep: {error}")

        time.sleep(0.002)


# =====================================================================
# MAIN
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DAQ Headless 4 nodos — sin GUI")
    parser.add_argument("--mode", choices=["simulation", "mscl"], default="simulation")
    parser.add_argument("--port", default=COM_PORT, help="Puerto serie para MSCL (ej: COM4)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  DAQ Headless — Modo: {args.mode.upper()}")
    print(f"  PID: {os.getpid()}")
    print(f"  Puerto: {args.port}")
    print(f"  Nodos: {NODE_ADDRESSES}")
    print(f"  HDF5 dir: {H5_ROOT_DIR}")
    print("=" * 60)

    flush_thread = threading.Thread(target=flush_pending_loop, daemon=True, name="hdf5-flush")
    flush_thread.start()

    try:
        if args.mode == "simulation":
            run_simulation()
        else:
            run_mscl(port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        global _running
        _running = False
        time.sleep(WRITE_FLUSH_INTERVAL + 0.2)
        print(f"\n[DAQ Headless] Finalizado. Total muestras: {_sample_count}")


if __name__ == '__main__':
    main()
