"""
essfile.py - ESSFile class for working with obs-period oriented ESS data

Provides the same API as the Tcl df::File class: event selection,
sparse/nested extraction, and stimdg access. Built on top of essread.py.

Usage:
    from essfile import ESSFile

    f = ESSFile('session.ess')
    print(f.identity)
    print(f.params)

    # Select events by name
    mask = f.select_evt('ENDTRIAL')          # list of bool arrays
    mask = f.select_evt('ENDTRIAL', 'CORRECT')  # with subtype

    # Sparse extraction (one value per trial, fill missing)
    valid = f.valid_obs(...)
    times = f.event_time_sparse(valid, 'RESP')
    params = f.event_param_sparse(valid, 'STIMTYPE')

    # Stimdg access
    f.stimdg['shape_id'][stimtype_indices]
"""

import numpy as np
from essfile.essread import read_ess


class ESSFile:
    """
    Wrapper around read_ess() output that provides the event query
    methods needed for trial extraction.

    Mirrors the Tcl df::File API.
    """

    def __init__(self, filename):
        self._ess = read_ess(filename)

        # Expose top-level attributes directly
        self.name = self._ess['name']
        self.version = self._ess['version']
        self.identity = self._ess['identity']
        self.params = self._ess['params']
        self.stimdg = self._ess['stimdg']
        self.n_obs = self._ess['n_obs']
        self.obs_times = self._ess['obs_times']
        self.extra_vars = self._ess['extra_vars']
        self.session_vars = self._ess['session_vars']
        self.pre_datapoints = self._ess['pre_datapoints']

        # Event name tables
        self._event_names = self._ess['event_names']
        self._name_to_id = self._ess['name_to_id']
        self._subtype_names = self._ess['subtype_names']
        self._subtype_ids = self._ess['subtype_ids']

        # Obs-period data (list of dicts)
        self._obs = self._ess['obs']

    # ---- name lookups ----

    def type_id(self, name):
        """Look up event type ID by name. e.g. type_id('ENDTRIAL') -> 40"""
        return self._name_to_id.get(name)

    def type_name(self, tid):
        """Look up event type name by ID."""
        return self._event_names.get(tid, f'type_{tid}')

    def subtype_id(self, type_name, subtype_name):
        """Look up subtype ID. e.g. subtype_id('ENDTRIAL', 'CORRECT') -> 1"""
        tid = self.type_id(type_name)
        if tid is None:
            return None
        st = self._subtype_names.get(tid, {})
        return st.get(subtype_name)

    def subtype_name(self, tid, sid):
        """Look up subtype name by IDs."""
        snames = self._subtype_ids.get(tid, {})
        return snames.get(sid, f'sub_{sid}')

    def has_event_type(self, type_name):
        """Check if event type exists in the name table."""
        return type_name in self._name_to_id

    def evt(self, type_name, subtype_name=None):
        """
        Resolve event name(s) to IDs.

        Returns type_id if no subtype given, (type_id, subtype_id) if subtype given.
        Accepts integer IDs as pass-through.
        """
        if isinstance(type_name, int):
            tid = type_name
        else:
            tid = self.type_id(type_name)
            if tid is None:
                return None

        if subtype_name is None:
            return tid

        if isinstance(subtype_name, int):
            sid = subtype_name
        else:
            sid = self.subtype_id(type_name, subtype_name)
            if sid is None:
                return None

        return tid, sid

    # ---- core event selection ----

    def select_evt(self, type_name, subtype_name=None):
        """
        Select events matching type (and optionally subtype) across all obs periods.

        Returns:
            list of np.array(bool), one per obs period.
            Each array is True where the event matches within that obs.

        Equivalent to Tcl: $f select_evt TYPE ?SUBTYPE?
        """
        tid = self.type_id(type_name) if isinstance(type_name, str) else type_name
        if tid is None:
            return [np.array([], dtype=bool) for _ in self._obs]

        result = []
        for obs in self._obs:
            type_match = obs['e_types'] == tid
            if subtype_name is not None:
                if isinstance(subtype_name, str):
                    sid = self.subtype_id(type_name, subtype_name)
                else:
                    sid = subtype_name
                if sid is not None:
                    type_match = type_match & (obs['e_subtypes'] == sid)
                else:
                    type_match = np.zeros(len(obs['e_types']), dtype=bool)
            result.append(type_match)
        return result

    # ---- per-obs accessors (apply mask) ----

    def event_times(self, mask):
        """Get event times filtered by mask. Returns list of arrays, one per obs."""
        return [obs['e_times'][m] for obs, m in zip(self._obs, mask)]

    def event_subtypes(self, mask):
        """Get event subtypes filtered by mask. Returns list of arrays, one per obs."""
        return [obs['e_subtypes'][m] for obs, m in zip(self._obs, mask)]

    def event_params(self, mask):
        """Get event params filtered by mask. Returns list of lists, one per obs."""
        return [[obs['e_params'][j] for j in np.where(m)[0]]
                for obs, m in zip(self._obs, mask)]

    # ---- values across all obs (unpacked, one per obs) ----

    def event_time_values(self, type_name, subtype_name=None):
        """Get one event time per obs period (first match). Assumes event exists in every obs."""
        mask = self.select_evt(type_name, subtype_name)
        return np.array([int(obs['e_times'][m][0]) if m.any() else -1
                         for obs, m in zip(self._obs, mask)], dtype=np.int32)

    def event_subtype_values(self, type_name, subtype_name=None):
        """Get one event subtype per obs period (first match)."""
        mask = self.select_evt(type_name, subtype_name)
        return np.array([int(obs['e_subtypes'][m][0]) if m.any() else -1
                         for obs, m in zip(self._obs, mask)], dtype=np.int32)

    def event_param_values(self, type_name, subtype_name=None):
        """Get one event param per obs period (first match)."""
        mask = self.select_evt(type_name, subtype_name)
        result = []
        for obs, m in zip(self._obs, mask):
            idxs = np.where(m)[0]
            if len(idxs) > 0:
                val = obs['e_params'][idxs[0]]
                # Unwrap single-element arrays to scalar
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                result.append(val)
            else:
                result.append(None)
        return result

    # ---- sparse extraction (handle missing events) ----

    def event_time_sparse(self, valid_indices, type_name, subtype_name=None, fill=-1):
        """
        Get event time for valid trials, filling missing with fill value.

        Args:
            valid_indices: np.array of int obs indices (0-based)
            type_name: event type name or ID
            subtype_name: optional subtype name or ID
            fill: value for obs periods where event didn't occur

        Returns:
            np.array(int32) of length len(valid_indices)
        """
        mask = self.select_evt(type_name, subtype_name)
        result = []
        for i in valid_indices:
            m = mask[i]
            if m.any():
                result.append(int(self._obs[i]['e_times'][m][0]))
            else:
                result.append(fill)
        return np.array(result, dtype=np.int32)

    def event_subtype_sparse(self, valid_indices, type_name, subtype_name=None, fill=-1):
        """Get event subtype for valid trials, filling missing with fill value."""
        mask = self.select_evt(type_name, subtype_name)
        result = []
        for i in valid_indices:
            m = mask[i]
            if m.any():
                result.append(int(self._obs[i]['e_subtypes'][m][0]))
            else:
                result.append(fill)
        return np.array(result, dtype=np.int32)

    def event_param_sparse(self, valid_indices, type_name, subtype_name=None, fill=-1):
        """
        Get event param for valid trials, filling missing with fill value.

        Returns list (not array) since params can be heterogeneous.
        For numeric params, returns unwrapped scalars.
        """
        mask = self.select_evt(type_name, subtype_name)
        result = []
        for i in valid_indices:
            m = mask[i]
            idxs = np.where(m)[0]
            if len(idxs) > 0:
                val = self._obs[i]['e_params'][idxs[0]]
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                result.append(val)
            else:
                result.append(fill)
        return result

    # ---- nested extraction (multiple events per trial) ----

    def event_times_nested(self, valid_indices, type_name, subtype_name=None):
        """Get event times as list of arrays (one per valid trial). For multi-occurrence events."""
        mask = self.select_evt(type_name, subtype_name)
        return [self._obs[i]['e_times'][mask[i]] for i in valid_indices]

    def event_params_nested(self, valid_indices, type_name, subtype_name=None):
        """Get event params as list of lists (one per valid trial)."""
        mask = self.select_evt(type_name, subtype_name)
        return [[self._obs[i]['e_params'][j] for j in np.where(mask[i])[0]]
                for i in valid_indices]

    def event_subtypes_nested(self, valid_indices, type_name, subtype_name=None):
        """Get event subtypes as list of arrays (one per valid trial)."""
        mask = self.select_evt(type_name, subtype_name)
        return [self._obs[i]['e_subtypes'][mask[i]] for i in valid_indices]

    # ---- has_event_occurrences ----

    def has_event_occurrences(self, type_name, subtype_name=None):
        """Check if event actually occurred in any obs period."""
        mask = self.select_evt(type_name, subtype_name)
        return any(m.any() for m in mask)

    # ---- extra var access ----

    def get_extra_var(self, varname, valid_indices):
        """Get extra variable data for valid trials."""
        if varname not in self.extra_vars:
            return None
        data = self.extra_vars[varname]
        return [data[i] for i in valid_indices]

    # ---- metadata ----

    @property
    def meta(self):
        """Return metadata dict (mirrors Tcl $f meta)."""
        return {
            'filepath': self.name,
            'subject': self.identity.get('subject', ''),
            'system': self.identity.get('ess', ''),
            'protocol': self.identity.get('protocol', ''),
            'variant': self.identity.get('variant', ''),
            'hostname': self.identity.get('hostname', ''),
            'n_obs': self.n_obs,
        }

    # ---- string representation ----

    def __repr__(self):
        return (f"ESSFile('{self.name}', n_obs={self.n_obs}, "
                f"subject='{self.identity.get('subject', '')}', "
                f"system='{self.identity.get('ess', '')}')")
