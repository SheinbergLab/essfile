"""
em.py - Eye movement analysis module

Provides eye tracking analysis utilities including:
  - Biquadratic calibration fitting and transformation
  - Calibration extraction from ESS session data
  - Raw data stream processing (x/y separation, P1-P4 computation)
  - Timestamp normalization

Python equivalent of em.tcl + biquadratic.tcl.

Usage:
    from essfile import ESSFile
    import em

    f = ESSFile('session.ess')

    # Extract calibration from session data
    calib = em.extract_calibration(f)

    # Process raw eye tracking streams
    eye_data = em.process_raw_streams(f, valid_indices, calibration=calib)
    # eye_data['em_h_deg'], eye_data['em_v_deg'] = calibrated position
"""

import numpy as np


# ======================================================================
#                    Biquadratic Calibration
# ======================================================================

def biquadratic_evaluate(coeffs, x, y):
    """
    Evaluate biquadratic polynomial at points (x, y).

    X = a₀ + a₁x + a₂y + a₃x² + a₄y² + a₅xy + a₆x²y + a₇xy² + a₈x²y²

    Args:
        coeffs: array-like of 9 coefficients [a₀..a₈]
        x, y: scalar or numpy arrays (broadcasting supported)

    Returns:
        Evaluated polynomial, same shape as x/y
    """
    a = np.asarray(coeffs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x2 = x * x
    y2 = y * y
    xy = x * y

    return (a[0] + a[1]*x + a[2]*y + a[3]*x2 + a[4]*y2
            + a[5]*xy + a[6]*x2*y + a[7]*x*y2 + a[8]*x2*y2)


def biquadratic_calibrate(x_coeffs, y_coeffs, h_raw, v_raw):
    """
    Apply biquadratic calibration to raw eye position data.

    Maps raw eye position (e.g., P1-P4 difference in pixels)
    to calibrated position (degrees visual angle).

    Args:
        x_coeffs: 9 coefficients for horizontal calibration
        y_coeffs: 9 coefficients for vertical calibration
        h_raw: raw horizontal position (scalar, array, or list of arrays)
        v_raw: raw vertical position (scalar, array, or list of arrays)

    Returns:
        (h_deg, v_deg) tuple, same structure as inputs
    """
    if isinstance(h_raw, list):
        # Per-trial nested data — apply to each trial
        h_deg = [biquadratic_evaluate(x_coeffs, h, v) for h, v in zip(h_raw, v_raw)]
        v_deg = [biquadratic_evaluate(y_coeffs, h, v) for h, v in zip(h_raw, v_raw)]
        return h_deg, v_deg
    else:
        h_deg = biquadratic_evaluate(x_coeffs, h_raw, v_raw)
        v_deg = biquadratic_evaluate(y_coeffs, h_raw, v_raw)
        return h_deg, v_deg


def biquadratic_fit(eye_x, eye_y, calib_x, calib_y):
    """
    Fit biquadratic mapping from eye position to calibration targets.

    Solves the least-squares system for 9-term biquadratic polynomial.

    Args:
        eye_x, eye_y: raw eye positions at calibration points
        calib_x, calib_y: known calibration target positions

    Returns:
        (x_coeffs, y_coeffs) tuple, each a 9-element array
    """
    eye_x = np.asarray(eye_x, dtype=np.float64)
    eye_y = np.asarray(eye_y, dtype=np.float64)
    calib_x = np.asarray(calib_x, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    n = len(eye_x)
    if n < 9:
        raise ValueError(f"Need at least 9 points for biquadratic fit, have {n}")

    # Design matrix: [1, x, y, x², y², xy, x²y, xy², x²y²]
    x, y = eye_x, eye_y
    A = np.column_stack([
        np.ones(n), x, y, x*x, y*y, x*y,
        x*x*y, x*y*y, x*x*y*y
    ])

    # Least squares solve
    x_coeffs, _, _, _ = np.linalg.lstsq(A, calib_x, rcond=None)
    y_coeffs, _, _, _ = np.linalg.lstsq(A, calib_y, rcond=None)

    return x_coeffs, y_coeffs


def biquadratic_rms(x_coeffs, y_coeffs, eye_x, eye_y, calib_x, calib_y):
    """
    Calculate RMS error of biquadratic fit.

    Returns:
        (rms_x, rms_y, rms_combined) tuple
    """
    pred_x = biquadratic_evaluate(x_coeffs, eye_x, eye_y)
    pred_y = biquadratic_evaluate(y_coeffs, eye_x, eye_y)

    err_x = np.asarray(calib_x) - pred_x
    err_y = np.asarray(calib_y) - pred_y

    rms_x = np.sqrt(np.mean(err_x**2))
    rms_y = np.sqrt(np.mean(err_y**2))
    rms_combined = np.sqrt((rms_x**2 + rms_y**2) / 2.0)

    return rms_x, rms_y, rms_combined


# ======================================================================
#                    Raw Data Stream Processing
# ======================================================================

def separate_xy(interleaved):
    """
    Separate interleaved x,y data into separate arrays.

    Eye tracking data often comes as [x0, y0, x1, y1, ...]

    Args:
        interleaved: array or list of arrays of interleaved x,y values

    Returns:
        (x, y) tuple — arrays or lists of arrays matching input structure
    """
    if isinstance(interleaved, list):
        # Per-trial nested data
        xs, ys = [], []
        for trial in interleaved:
            if trial is not None and len(trial) >= 2:
                arr = np.asarray(trial)
                xs.append(arr[0::2])
                ys.append(arr[1::2])
            else:
                xs.append(np.array([], dtype=np.float32))
                ys.append(np.array([], dtype=np.float32))
        return xs, ys
    else:
        arr = np.asarray(interleaved)
        return arr[0::2], arr[1::2]


def compute_p1p4(p1_x, p1_y, p4_x, p4_y):
    """
    Compute P1-P4 difference (dual Purkinje eye position).

    Returns:
        (h_diff, v_diff) tuple — same structure as inputs
    """
    if isinstance(p1_x, list):
        h = [np.asarray(a) - np.asarray(b) for a, b in zip(p1_x, p4_x)]
        v = [np.asarray(a) - np.asarray(b) for a, b in zip(p1_y, p4_y)]
        return h, v
    else:
        return np.asarray(p1_x) - np.asarray(p4_x), np.asarray(p1_y) - np.asarray(p4_y)


def normalize_timestamps(timestamps):
    """
    Normalize timestamps to seconds from first sample of each trial.

    Args:
        timestamps: list of arrays (one per trial)

    Returns:
        list of arrays, relative to first sample
    """
    result = []
    for ts in timestamps:
        ts = np.asarray(ts, dtype=np.float64)
        if len(ts) > 0:
            result.append(ts - ts[0])
        else:
            result.append(ts)
    return result


def compute_min_lengths(streams):
    """
    Compute minimum sample count across multiple data streams per trial.

    Args:
        streams: dict of (data, is_interleaved) pairs
                 e.g. {'pupil': (data, True), 'time': (data, False)}

    Returns:
        array of minimum lengths per trial
    """
    all_lengths = []
    for data, is_interleaved in streams.values():
        lengths = []
        for trial in data:
            if trial is not None:
                n = len(trial)
                if is_interleaved:
                    n //= 2
                lengths.append(n)
            else:
                lengths.append(0)
        all_lengths.append(lengths)

    return np.min(all_lengths, axis=0).astype(int)


def truncate_to_length(data, ns, is_interleaved=False):
    """
    Truncate per-trial data to specified lengths.

    Args:
        data: list of arrays (one per trial)
        ns: array of target lengths per trial
        is_interleaved: if True, multiply length by 2

    Returns:
        list of truncated arrays
    """
    result = []
    for trial, n in zip(data, ns):
        if trial is not None and n > 0:
            end = n * 2 if is_interleaved else n
            result.append(np.asarray(trial)[:end])
        else:
            result.append(np.array([]))
    return result


# ======================================================================
#                    Calibration Extraction
# ======================================================================

def extract_calibration(f):
    """
    Extract biquadratic calibration from an ESSFile.

    Searches pre_datapoints first (calibration set before obs periods),
    then session_vars as fallback.

    Args:
        f: ESSFile object

    Returns:
        dict with 'x_coeffs' and 'y_coeffs' (each 9-element list),
        plus metadata (source, rms_error, etc.)
        Returns None if no calibration found.
    """
    raw = None

    # Search pre_datapoints first (most common location)
    if hasattr(f, 'pre_datapoints'):
        for key, val in f.pre_datapoints.items():
            if 'em/biquadratic' in key or 'em:biquadratic' in key:
                raw = val
                break

    # Fall back to session_vars
    if raw is None:
        for key, val in f.session_vars.items():
            if 'em/biquadratic' in key or 'em:biquadratic' in key:
                raw = val
                break

    if raw is None:
        return None

    # Unwrap list if needed
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None

    # Parse the calibration data
    # This depends on how it's serialized — could be a Tcl dict string
    # or structured data. Handle both cases.
    if isinstance(raw, str):
        calib = _parse_tcl_dict(raw)
    elif isinstance(raw, dict):
        calib = raw
    else:
        return None

    # Validate
    if 'x_coeffs' not in calib or 'y_coeffs' not in calib:
        return None

    # Ensure coefficients are numeric arrays
    if isinstance(calib['x_coeffs'], str):
        calib['x_coeffs'] = [float(x) for x in calib['x_coeffs'].split()]
    if isinstance(calib['y_coeffs'], str):
        calib['y_coeffs'] = [float(x) for x in calib['y_coeffs'].split()]

    return calib


def calibration_coeffs(calib):
    """
    Get just the (x_coeffs, y_coeffs) pair from a calibration dict.

    Args:
        calib: dict from extract_calibration()

    Returns:
        (x_coeffs, y_coeffs) tuple, or (None, None) if calib is None
    """
    if calib is None:
        return None, None
    return calib['x_coeffs'], calib['y_coeffs']


def calibration_info(calib):
    """Print calibration provenance for diagnostics."""
    if calib is None:
        print("em::calibration: none")
        return

    source = calib.get('source', 'unknown')
    rms = calib.get('rms_error', '?')
    n = calib.get('n_trials', '?')
    if isinstance(rms, (int, float)):
        rms = f"{rms:.4f}"
    print(f"em::calibration: source={source} rms={rms}deg n_trials={n}")


# ======================================================================
#                    Process Raw Streams (main entry point)
# ======================================================================

def process_raw_streams(f, valid_indices, calibration=None):
    """
    Process raw eye tracking streams into standard format.

    Takes raw em data streams from extra_vars, separates x/y,
    optionally applies biquadratic calibration, and returns
    a dict of processed arrays.

    Args:
        f: ESSFile object
        valid_indices: array of valid obs indices
        calibration: dict from extract_calibration(), or None

    Returns:
        dict with keys (present only if source data exists):
            pupil_x, pupil_y: pupil center position per trial
            p1_x, p1_y: first Purkinje image position
            p4_x, p4_y: fourth Purkinje image position
            eye_raw_h, eye_raw_v: raw eye signal (as used for calibration)
            em_h_deg, em_v_deg: calibrated position in degrees (if calibration provided)
            em_time: timestamps per trial
            em_seconds: normalized timestamps (from trial start)
            pupil_r: pupil radius
            in_blink: blink detection flag
            frame_id: camera frame IDs
    """
    result = {}

    # Map of extra var names to (output_key, is_interleaved)
    stream_map = {
        'em/pupil':         ('pupil', True),
        'em/p1':            ('p1', True),
        'em/p4':            ('p4', True),
        'eyetracking/raw':  ('eye_raw', True),
        'em/time':          ('time', False),
        'em/pupil_r':       ('pupil_r', False),
        'em/blink':         ('blink', False),
        'em/frame_id':      ('frame_id', False),
    }

    # Collect available streams
    available = {}
    for var_name, (key, interleaved) in stream_map.items():
        data = f.get_extra_var(var_name, valid_indices)
        if data is not None and any(d is not None for d in data):
            # Concatenate chunked data: each trial may have multiple
            # datapoint updates (list of arrays) that need merging
            merged = []
            for trial in data:
                if trial is None:
                    merged.append(np.array([]))
                elif isinstance(trial, list):
                    # Multiple updates per trial — concatenate
                    arrays = [np.asarray(a) for a in trial if a is not None]
                    merged.append(np.concatenate(arrays) if arrays else np.array([]))
                else:
                    merged.append(np.asarray(trial))
            available[key] = (merged, interleaved)

    if not available:
        result['_consumed_vars'] = []
        return result

    # Compute minimum lengths across all streams for consistent truncation
    ns = compute_min_lengths(available)

    # Process interleaved streams (separate x/y)
    for key in ['pupil', 'p1', 'p4']:
        if key in available:
            data, _ = available[key]
            truncated = truncate_to_length(data, ns, is_interleaved=True)
            x, y = separate_xy(truncated)
            result[f'{key}_x'] = x
            result[f'{key}_y'] = y

    # Process eye_raw and apply calibration
    if 'eye_raw' in available:
        data, _ = available['eye_raw']
        truncated = truncate_to_length(data, ns, is_interleaved=True)
        h_raw, v_raw = separate_xy(truncated)
        result['eye_raw_h'] = h_raw
        result['eye_raw_v'] = v_raw

        if calibration is not None:
            x_coeffs, y_coeffs = calibration_coeffs(calibration)
            if x_coeffs is not None:
                h_deg, v_deg = biquadratic_calibrate(x_coeffs, y_coeffs, h_raw, v_raw)
                result['em_h_deg'] = h_deg
                result['em_v_deg'] = v_deg

    # Process scalar streams
    if 'time' in available:
        data, _ = available['time']
        truncated = truncate_to_length(data, ns, is_interleaved=False)
        result['em_time'] = truncated
        result['em_seconds'] = normalize_timestamps(truncated)

    for key in ['pupil_r', 'blink', 'frame_id']:
        if key in available:
            data, _ = available[key]
            truncated = truncate_to_length(data, ns, is_interleaved=False)
            out_key = 'in_blink' if key == 'blink' else key
            result[out_key] = truncated

    # Track which extra var names were consumed by this processing
    result['_consumed_vars'] = list(stream_map.keys())

    return result


# ======================================================================
#                    Tcl Dict Parser (utility)
# ======================================================================

def _parse_tcl_dict(s):
    """
    Parse a simple Tcl dict string into a Python dict.

    Handles space-separated key-value pairs where values may be
    brace-quoted. This is not a full Tcl parser but handles the
    common calibration dict format.
    """
    result = {}
    s = s.strip()
    if not s:
        return result

    i = 0
    tokens = []

    while i < len(s):
        # Skip whitespace
        while i < len(s) and s[i] in ' \t\n':
            i += 1
        if i >= len(s):
            break

        # Brace-quoted value
        if s[i] == '{':
            depth = 1
            i += 1
            start = i
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                i += 1
            tokens.append(s[start:i-1])
        else:
            # Unquoted word
            start = i
            while i < len(s) and s[i] not in ' \t\n':
                i += 1
            tokens.append(s[start:i])

    # Pair up as key-value
    for j in range(0, len(tokens) - 1, 2):
        result[tokens[j]] = tokens[j + 1]

    return result
