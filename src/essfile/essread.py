"""
essread.py - Pure Python reader for dserv .ess log files

Reads the binary 'dslog' format produced by dserv's ess_ds_logger
and returns structured Python data (dicts of numpy arrays/lists).

Dependencies: numpy (required), dgread (optional, for stimdg decoding)
"""

import struct
import numpy as np
from pathlib import Path

__version__ = "0.2.0"

# ---- dslog constants ----

DSLOG_HEADER_SIZE = 16
DSLOG_MAGIC = b'dslog'

# ds_datatype_t enum values
DSERV_BYTE = 0
DSERV_STRING = 1
DSERV_FLOAT = 2
DSERV_DOUBLE = 3
DSERV_SHORT = 4
DSERV_INT = 5
DSERV_DG = 6
DSERV_SCRIPT = 7
DSERV_TRIGGER_SCRIPT = 8
DSERV_EVT = 9
DSERV_NONE = 10
DSERV_UNKNOWN = 11

# Well-known event type IDs (from standard ESS event table)
# These are the numeric IDs; the name table confirms them at runtime
EVT_NAME = 1        # event type name definitions
EVT_PARAM = 5       # parameter name/value pairs
EVT_SUBTYPES = 6    # subtype name definitions
EVT_SYSTEM_STATE = 7
EVT_TIME = 8        # epoch timestamp
EVT_ID = 18         # session identity (system, subject, etc.)
EVT_BEGINOBS = 19
EVT_ENDOBS = 20

# Subtype indices for EVT_ID
ID_ESS = 0
ID_SUBJECT = 1
ID_PROTOCOL = 2
ID_VARIANT = 3
ID_HOSTNAME = 4


# ---- low-level binary reading ----

# Pre-compiled struct formats
_S_UINT16 = struct.Struct('<H')
_S_UINT32 = struct.Struct('<I')
_S_UINT64 = struct.Struct('<Q')
_S_BODY = struct.Struct('<QI')    # timestamp(8) + flags(4)


def _read_header(fp):
    """Read and validate dslog header. Returns (version, timestamp_us)."""
    header = fp.read(DSLOG_HEADER_SIZE)
    if len(header) < DSLOG_HEADER_SIZE:
        raise ValueError("File too short to contain dslog header")
    if header[:5] != DSLOG_MAGIC:
        raise ValueError("Not a dslog file (bad magic)")
    version = header[5]
    if version == 0:
        raise ValueError("Invalid dslog version 0")
    timestamp_us = _S_UINT64.unpack_from(header, 8)[0]
    return version, timestamp_us


def _read_datapoint(fp):
    """
    Read one datapoint record from the file.

    Returns dict with keys:
        varname, timestamp, flags, dtype, is_event,
        event_type, event_subtype, event_puttype,
        data_len, data

    Returns None at EOF.
    """
    raw = fp.read(2)
    if len(raw) < 2:
        return None

    varlen = _S_UINT16.unpack(raw)[0]
    varname = fp.read(varlen).decode('ascii', errors='replace')

    raw = fp.read(20)  # timestamp(8) + flags(4) + type_union(4) + data_len(4)
    if len(raw) < 20:
        return None

    timestamp_us, flags = _S_BODY.unpack_from(raw, 0)

    type_bytes = raw[12:16]
    dtype = type_bytes[0]
    data_len = _S_UINT32.unpack_from(raw, 16)[0]

    data = fp.read(data_len) if data_len > 0 else b''
    if len(data) < data_len:
        return None

    dp = {
        'varname': varname,
        'timestamp': timestamp_us,
        'flags': flags,
        'data_len': data_len,
        'data': data,
    }

    if dtype == DSERV_EVT:
        dp['is_event'] = True
        dp['dtype'] = DSERV_EVT
        dp['event_type'] = type_bytes[1]
        dp['event_subtype'] = type_bytes[2]
        dp['event_puttype'] = type_bytes[3]
    else:
        dp['is_event'] = False
        dp['dtype'] = _S_UINT32.unpack_from(type_bytes, 0)[0]
        dp['event_type'] = 0
        dp['event_subtype'] = 0
        dp['event_puttype'] = 0

    return dp


def _decode_value(dtype, data):
    """Decode raw bytes into a Python/numpy value based on dserv type."""
    if len(data) == 0:
        return None

    if dtype == DSERV_BYTE:
        return np.frombuffer(data, dtype=np.uint8)
    elif dtype == DSERV_STRING:
        return data.decode('ascii', errors='replace')
    elif dtype == DSERV_FLOAT:
        return np.frombuffer(data, dtype=np.float32)
    elif dtype == DSERV_DOUBLE:
        return np.frombuffer(data, dtype=np.float64)
    elif dtype == DSERV_SHORT:
        return np.frombuffer(data, dtype=np.int16)
    elif dtype == DSERV_INT:
        return np.frombuffer(data, dtype=np.int32)
    elif dtype == DSERV_DG:
        return data  # raw DG bytes
    elif dtype == DSERV_NONE:
        return None
    else:
        return data


# ---- public API: flat reader ----

def read_dslog(filename):
    """
    Read a dslog file and return column-oriented data,
    matching the structure of dslog_to_dg().

    Returns:
        dict with:
            'version':   int
            'varname':   list of str  (per datapoint)
            'timestamp': list of float (seconds from file start)
            'vals':      list of decoded values
    """
    fp = open(filename, 'rb')
    version, start_ts = _read_header(fp)
    start_sec = start_ts / 1_000_000.0

    varnames = ['logger:open']
    timestamps = [0.0]
    vals = [version]

    while True:
        dp = _read_datapoint(fp)
        if dp is None:
            break

        time_sec = dp['timestamp'] / 1_000_000.0 - start_sec

        if dp['is_event']:
            name = f"evt:{dp['event_type']}:{dp['event_subtype']}"
            value_dtype = dp['event_puttype']
        else:
            name = dp['varname']
            value_dtype = dp['dtype']

        varnames.append(name)
        timestamps.append(time_sec)
        vals.append(_decode_value(value_dtype, dp['data']))

    fp.close()

    return {
        'version': version,
        'varname': varnames,
        'timestamp': timestamps,
        'vals': vals,
    }


# ---- pre-obs parsing ----

def _parse_preamble(dslog):
    """
    Parse the pre-obs section of a dslog to extract:
      - stimdg (decoded via dgread if available, raw bytes otherwise)
      - event type names:   {type_id: name_str}
      - subtype names:      {type_id: {name_str: subtype_id}}
      - session identity:   {ess, subject, protocol, variant, hostname}
      - parameters:         {name: value_str}

    Args:
        dslog: dict from read_dslog()

    Returns:
        dict with keys: stimdg, stimdg_raw, event_names, name_to_id,
                        subtype_names, subtype_ids, identity, params,
                        first_obs_index
    """
    varnames = dslog['varname']
    vals = dslog['vals']

    # Find first BeginObs to bound the preamble
    first_obs = None
    for i, name in enumerate(varnames):
        if name == f'evt:{EVT_BEGINOBS}:0':
            first_obs = i
            break

    if first_obs is None:
        first_obs = len(varnames)

    # ---- stimdg ----
    stimdg_raw = None
    stimdg = None
    for i in range(first_obs):
        if varnames[i] == 'stimdg' and isinstance(vals[i], bytes):
            stimdg_raw = vals[i]
            break

    if stimdg_raw is not None:
        try:
            import dgread
            stimdg = dgread.fromString(stimdg_raw)
        except ImportError:
            stimdg = stimdg_raw  # keep raw bytes as fallback
        except Exception:
            stimdg = stimdg_raw

    # ---- event type names (evt:1:N -> name string) ----
    event_names = {}     # {type_id: name}
    name_to_id = {}      # {name: type_id} reverse lookup
    for i in range(first_obs):
        if varnames[i].startswith(f'evt:{EVT_NAME}:'):
            type_id = int(varnames[i].split(':')[2])
            name_str = vals[i]
            if isinstance(name_str, str):
                event_names[type_id] = name_str
                name_to_id[name_str] = type_id

    # ---- subtype names (evt:6:N -> "NAME0 0 NAME1 1 ...") ----
    subtype_names = {}   # {type_id: {name: subtype_id}}
    subtype_ids = {}     # {type_id: {subtype_id: name}}
    for i in range(first_obs):
        if varnames[i].startswith(f'evt:{EVT_SUBTYPES}:'):
            type_id = int(varnames[i].split(':')[2])
            text = vals[i]
            if isinstance(text, str):
                parts = text.split()
                names_map = {}
                ids_map = {}
                for j in range(0, len(parts) - 1, 2):
                    sname = parts[j]
                    sid = int(parts[j + 1])
                    names_map[sname] = sid
                    ids_map[sid] = sname
                subtype_names[type_id] = names_map
                subtype_ids[type_id] = ids_map

    # ---- session identity (evt:18:N) ----
    identity = {}
    id_fields = {ID_ESS: 'ess', ID_SUBJECT: 'subject',
                 ID_PROTOCOL: 'protocol', ID_VARIANT: 'variant',
                 ID_HOSTNAME: 'hostname'}
    for i in range(first_obs):
        if varnames[i].startswith(f'evt:{EVT_ID}:'):
            sub_id = int(varnames[i].split(':')[2])
            if sub_id in id_fields and isinstance(vals[i], str):
                identity[id_fields[sub_id]] = vals[i]

    # ---- parameters (evt:5:0 = name, evt:5:1 = value, alternating) ----
    params = {}
    pending_name = None
    for i in range(first_obs):
        if varnames[i] == f'evt:{EVT_PARAM}:0':
            pending_name = vals[i] if isinstance(vals[i], str) else None
        elif varnames[i] == f'evt:{EVT_PARAM}:1':
            if pending_name is not None:
                params[pending_name] = vals[i] if isinstance(vals[i], str) else str(vals[i])
                pending_name = None

    # ---- pre-obs datapoints (non-event, non-stimdg) ----
    # These include things like em/biquadratic calibration data
    # that are set before the first obs period begins
    pre_datapoints = {}
    for i in range(first_obs):
        name = varnames[i]
        if name.startswith('evt:') or name in ('stimdg', 'logger:open'):
            continue
        pre_datapoints[name] = vals[i]

    return {
        'stimdg': stimdg,
        'stimdg_raw': stimdg_raw,
        'event_names': event_names,
        'name_to_id': name_to_id,
        'subtype_names': subtype_names,
        'subtype_ids': subtype_ids,
        'identity': identity,
        'params': params,
        'pre_datapoints': pre_datapoints,
        'first_obs_index': first_obs,
    }


# ---- public API: obs-period oriented reader ----

def read_ess(filename):
    """
    Read an ESS log file and return obs-period oriented data.

    Builds on read_dslog(), parses the preamble for event name tables
    and session metadata, then segments the datapoint stream into
    observation periods.

    Returns:
        dict with:
            'name':       str (derived from filename)
            'version':    int
            'identity':   dict (ess, subject, protocol, variant, hostname)
            'params':     dict (parameter name -> value string)
            'stimdg':     dict (from dgread) or raw bytes

            'event_names':   dict {type_id: name}
            'name_to_id':    dict {name: type_id}
            'subtype_names': dict {type_id: {name: subtype_id}}
            'subtype_ids':   dict {type_id: {subtype_id: name}}

            'n_obs':      int
            'obs_times':  list of int (ms from first obs)
            'obs':        list of obs-period dicts, each containing:
                'e_types':    np.array int32 (event type ids)
                'e_subtypes': np.array int32 (event subtype ids)
                'e_times':    np.array int32 (ms from obs start)
                'e_params':   list of decoded values
            'extra_vars': dict {varname: list of per-obs values}
            'session_vars': dict {varname: list of values}
    """
    # Step 1: flat read
    dslog = read_dslog(filename)

    # Step 2: parse preamble
    pre = _parse_preamble(dslog)

    stem = Path(filename).stem
    varnames = dslog['varname']
    timestamps = dslog['timestamp']
    vals = dslog['vals']

    # Event name strings for BeginObs/EndObs
    beginobs_prefix = f'evt:{EVT_BEGINOBS}:'
    endobs_prefix = f'evt:{EVT_ENDOBS}:'

    def _is_event(name):
        return name.startswith('evt:')

    def _parse_evt(name):
        """Parse 'evt:TYPE:SUBTYPE' -> (type_id, subtype_id)"""
        parts = name.split(':')
        return int(parts[1]), int(parts[2])

    # Known special datapoint varnames to skip
    special_vars = {'stimdg', 'eventlog/names', 'eventlog/events',
                    'ain/vals', 'logger:beginobs', 'logger:endobs',
                    'logger:open'}

    # Step 3: discover extra var names (inside obs) and session var names (outside)
    extra_varnames = []
    session_varnames = []
    getting_trial = False

    for i in range(pre['first_obs_index'], len(varnames)):
        name = varnames[i]

        if name.startswith(beginobs_prefix):
            getting_trial = True
            continue
        if name.startswith(endobs_prefix):
            getting_trial = False
            continue
        if _is_event(name) or name in special_vars:
            continue

        # Skip DG-typed values (raw bytes from embedded DGs)
        v = vals[i]
        if isinstance(v, bytes) and not isinstance(v, np.ndarray):
            continue

        if getting_trial:
            if name not in extra_varnames:
                extra_varnames.append(name)
        else:
            if name not in session_varnames:
                session_varnames.append(name)

    # Step 4: segment into obs periods
    obs_list = []
    obs_times = []
    extra_vars = {v: [] for v in extra_varnames}
    session_vars = {v: [] for v in session_varnames}

    first_obs_ts = None
    getting_trial = False

    cur_types = []
    cur_subtypes = []
    cur_times = []
    cur_params = []
    cur_extra = {v: [] for v in extra_varnames}
    cur_obs_start_ts = None

    def _finish_obs():
        nonlocal cur_types, cur_subtypes, cur_times, cur_params, cur_extra

        obs_list.append({
            'e_types': np.array(cur_types, dtype=np.int32),
            'e_subtypes': np.array(cur_subtypes, dtype=np.int32),
            'e_times': np.array(cur_times, dtype=np.int32),
            'e_params': cur_params,
        })

        for v in extra_varnames:
            extra_vars[v].append(cur_extra[v] if cur_extra[v] else None)

        cur_types = []
        cur_subtypes = []
        cur_times = []
        cur_params = []
        cur_extra = {v: [] for v in extra_varnames}

    for i in range(pre['first_obs_index'], len(varnames)):
        name = varnames[i]
        val = vals[i]
        ts = timestamps[i]

        if name.startswith(beginobs_prefix):
            if getting_trial:
                _finish_obs()

            if first_obs_ts is None:
                first_obs_ts = ts

            cur_obs_start_ts = ts
            getting_trial = True
            cur_types = []
            cur_subtypes = []
            cur_times = []
            cur_params = []
            cur_extra = {v: [] for v in extra_varnames}
            continue

        if getting_trial and _is_event(name):
            type_id, subtype_id = _parse_evt(name)
            evtime_ms = round((ts - cur_obs_start_ts) * 1000)
            cur_types.append(type_id)
            cur_subtypes.append(subtype_id)
            cur_times.append(evtime_ms)
            cur_params.append(val)

            if type_id == EVT_ENDOBS:
                obs_time_ms = round((cur_obs_start_ts - first_obs_ts) * 1000)
                obs_times.append(obs_time_ms)
                _finish_obs()
                getting_trial = False
            continue

        if not _is_event(name) and name not in special_vars:
            if getting_trial:
                if name in cur_extra:
                    cur_extra[name].append(val)
            else:
                if name in session_vars:
                    session_vars[name].append(val)

    return {
        'name': stem,
        'version': dslog['version'],
        'identity': pre['identity'],
        'params': pre['params'],
        'stimdg': pre['stimdg'],
        'pre_datapoints': pre['pre_datapoints'],

        'event_names': pre['event_names'],
        'name_to_id': pre['name_to_id'],
        'subtype_names': pre['subtype_names'],
        'subtype_ids': pre['subtype_ids'],

        'n_obs': len(obs_list),
        'obs_times': obs_times,
        'obs': obs_list,
        'extra_vars': extra_vars,
        'session_vars': session_vars,
    }


# ---- helper methods for working with read_ess() output ----

def event_type_id(ess, name):
    """Look up event type ID by name. e.g. event_type_id(ess, 'ENDTRIAL') -> 40"""
    return ess['name_to_id'].get(name)


def subtype_id(ess, type_name, subtype_name):
    """Look up subtype ID. e.g. subtype_id(ess, 'ENDTRIAL', 'CORRECT') -> 1"""
    type_id = event_type_id(ess, type_name)
    if type_id is None:
        return None
    st = ess['subtype_names'].get(type_id, {})
    return st.get(subtype_name)


def event_name(ess, type_id, sub_id=None):
    """Look up event name by ID. Optionally include subtype name."""
    tname = ess['event_names'].get(type_id, f'type_{type_id}')
    if sub_id is not None:
        snames = ess['subtype_ids'].get(type_id, {})
        sname = snames.get(sub_id, f'sub_{sub_id}')
        return f"{tname}:{sname}"
    return tname


def obs_events(ess, obs_index):
    """
    Get events for an obs period as a readable list of
    (time_ms, type_name:subtype_name, param).
    """
    obs = ess['obs'][obs_index]
    result = []
    for j in range(len(obs['e_types'])):
        tid = int(obs['e_types'][j])
        sid = int(obs['e_subtypes'][j])
        tms = int(obs['e_times'][j])
        param = obs['e_params'][j]
        name = event_name(ess, tid, sid)
        result.append((tms, name, param))
    return result


# ---- convenience ----

def summary(ess):
    """Print a summary of data returned by read_ess()."""
    print(f"File: {ess['name']}")
    print(f"Version: {ess['version']}")

    if ess['identity']:
        id_parts = []
        for k in ['ess', 'subject', 'protocol', 'variant']:
            if k in ess['identity']:
                id_parts.append(f"{k}={ess['identity'][k]}")
        print(f"Identity: {', '.join(id_parts)}")

    print(f"Obs periods: {ess['n_obs']}")

    if ess['params']:
        print(f"Parameters: {len(ess['params'])} settings")

    if ess['stimdg'] is not None:
        if isinstance(ess['stimdg'], dict):
            keys = list(ess['stimdg'].keys())
            n = len(next(iter(ess['stimdg'].values()))) if keys else 0
            print(f"Stimdg: {len(keys)} columns, {n} rows")
        else:
            print(f"Stimdg: {len(ess['stimdg'])} bytes (raw)")

    if ess['extra_vars']:
        active = [k for k, v in ess['extra_vars'].items()
                  if any(x is not None for x in v)]
        if active:
            print(f"Extra vars: {', '.join(active)}")

    if ess['session_vars']:
        active = [k for k, v in ess['session_vars'].items() if v]
        if active:
            print(f"Session vars: {', '.join(active)}")

    if ess['obs']:
        obs0 = ess['obs'][0]
        print(f"Obs 0: {len(obs0['e_types'])} events")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.ess>")
        sys.exit(1)

    ess = read_ess(sys.argv[1])
    summary(ess)

    # Show first obs period events with names
    if ess['obs']:
        print(f"\nObs 0 events:")
        for tms, name, param in obs_events(ess, 0):
            pstr = f"  {param}" if param is not None else ""
            print(f"  {tms:6d} ms  {name}{pstr}")
