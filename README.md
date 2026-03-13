# essfile

Pure Python reader and analysis tools for dserv ESS experiment log files.

## Installation

```bash
pip install essfile
```

This installs `essfile` with `numpy` and `dgread` (for stimulus parameter decoding).

For pandas DataFrame support:

```bash
pip install essfile[pandas]
```

## Quick Start

### Inspect an ESS file from the command line

```bash
essread session.ess
essread session.ess --raw   # show flat datapoint stream
```

### Read in Python

```python
from essfile import ESSFile

f = ESSFile('session.ess')
print(f.identity)    # {'ess': 'hapticvis', 'subject': 'human', ...}
print(f.params)      # {'stim_duration': '30000', ...}
print(f.n_obs)       # 12
print(f.stimdg)      # dict of numpy arrays (stimulus parameters)
```

### Extract trials

```python
from essfile import ESSFile
from essfile.extract.hapticvis import extract_trials

f = ESSFile('session.ess')
trials = extract_trials(f)

print(trials['rt'])       # reaction times
print(trials['correct'])  # 0/1 accuracy

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(trials)
```

### Low-level access

```python
from essfile import read_dslog, read_ess

# Flat datapoint stream (varname, timestamp, vals columns)
d = read_dslog('session.ess')

# Obs-period oriented with parsed preamble
ess = read_ess('session.ess')
```

## Architecture

The package has three layers:

| Layer | Function | Description |
|-------|----------|-------------|
| `essread` | `read_dslog()` | Binary parser → flat datapoint stream |
| `essread` | `read_ess()` | Segments into obs periods, parses preamble |
| `essfile` | `ESSFile` | Event query API (select, sparse, nested) |
| `extract.*` | `extract_trials()` | System-specific trial extraction |

### ESSFile API

The `ESSFile` class provides methods matching the Tcl `df::File` API:

```python
f = ESSFile('session.ess')

# Event selection (returns list of bool arrays, one per obs)
mask = f.select_evt('ENDTRIAL')
mask = f.select_evt('STIMULUS', 'ON')

# Sparse extraction (one value per trial, fill=-1 if missing)
valid = np.where(some_condition)[0]
times = f.event_time_sparse(valid, 'RESP')
params = f.event_param_sparse(valid, 'STIMTYPE')
subtypes = f.event_subtype_sparse(valid, 'ENDTRIAL')

# Nested extraction (variable count per trial)
decide_times = f.event_times_nested(valid, 'DECIDE', 'SELECT')

# Name lookups
f.type_id('ENDTRIAL')                    # -> 40
f.subtype_id('ENDTRIAL', 'CORRECT')      # -> 1
f.has_event_type('CHOICES')               # -> True
```

## Writing Extractors

System-specific extractors live in `essfile.extract`. Each provides an
`extract_trials(f)` function that takes an `ESSFile` and returns a dict
of equal-length arrays (one entry per valid trial).

See `essfile/extract/hapticvis.py` for a complete example.

## File Format

ESS files use the `dslog` binary format produced by dserv's logger:

- 16-byte header: magic `dslog`, version, timestamp
- Sequence of datapoint records: varname, timestamp, type, data
- Events encoded with type/subtype/puttype in a 4-byte union
- Pre-obs preamble contains event name tables, parameters, stimdg

## Requirements

- Python ≥ 3.9
- numpy
- dgread (for stimulus parameter decoding)
- pandas (optional, for DataFrame conversion)

## License

MIT
